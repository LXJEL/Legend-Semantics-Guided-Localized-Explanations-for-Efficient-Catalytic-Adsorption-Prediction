import json
import math
import os
import torch
from torch import nn
from AdslabData import DataPreFetcher
from tqdm import tqdm

USE_PER_ATOM_LOSS = False


class ModelEMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                new = param.detach()
                self.shadow[name].mul_(self.decay).add_(new, alpha=1.0 - self.decay)

    def store(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.detach().clone()

    def copy_to(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}


class Framework:
    def __init__(self, config):
        self.config = config
        self.train_loss = nn.L1Loss()
        self._distributed = getattr(config, "distributed", False)
        self._global_rank = getattr(config, "global_rank", 0)
        custom_subdir = getattr(config, "save_subdir", None)
        active_save_dir = getattr(config, "checkpoint_active_dir", None)
        active_log_dir = getattr(config, "log_active_dir", None)
        default_dir = "baseline"
        log_name = config.log_save_name.replace(".json", "_baseline.json")

        subdir = custom_subdir or default_dir
        self.save_dir = active_save_dir or os.path.join(config.checkpoint, subdir)
        self.log_dir = active_log_dir or os.path.join(config.log_dir, subdir)

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, log_name)

        self.pruning_log_path = None

        self._capture_geom = bool(getattr(config, "capture_geom_importance", False))
        self._geom_dump_count = max(1, int(getattr(config, "geom_importance_dump_count", 3)))
        self._geom_dump_dir = None
        self._geom_epoch_samples = []
        self._geom_epoch_index = None
        self._geom_epoch_flushed = False
        if self._capture_geom and self._is_main_process():
            active_log_dir = getattr(config, "log_active_dir", self.log_dir)
            self._geom_dump_dir = os.path.join(active_log_dir, "geom_samples")
            os.makedirs(self._geom_dump_dir, exist_ok=True)

        if self._is_main_process() and getattr(config, "mode", "train") != "val":
            print(f"[] Baseline")
            print(f"[] {self.save_dir}")
            print(f"[] {self.log_path}")
        self._last_stoken_sorted = None
        self._export_pruned_cif_sids = list(getattr(config, "export_pruned_cif_sids", []) or [])
        self._export_pruned_cif_mode = str(getattr(config, "export_pruned_cif_mode", "off") or "off").lower()
        self._cif_dump_dir = None
        self._cif_epoch_exported = set()
        self._cif_warned_missing = False
        if (
            self._export_pruned_cif_sids
            and self._export_pruned_cif_mode in {"train", "val", "both"}
            and self._is_main_process()
        ):
            export_dir = getattr(config, "export_pruned_cif_dir", None)
            if not export_dir:
                export_dir = os.path.join(self.log_dir, "cif_samples")
            self._cif_dump_dir = export_dir
            os.makedirs(self._cif_dump_dir, exist_ok=True)

        self._export_gradcam_sids = list(getattr(config, "export_gradcam_sids", []) or [])
        self._export_gradcam_all = bool(getattr(config, "export_gradcam_all", False))
        self._gradcam_dump_dir = None
        self._gradcam_warned_missing = False
        self._gradcam_epoch_exported = set()
        self._gradcam_first_n = max(0, int(getattr(config, "export_gradcam_first_n", 0)))
        self._gradcam_export_count = {}
        self._gradcam_export_cif_keep = bool(getattr(config, "export_gradcam_cif_keep", False))
        self._gradcam_export_scores = bool(getattr(config, "export_gradcam_scores", False))
        methods_raw = str(getattr(config, "export_gradcam_methods", "gnn,llm") or "")
        methods = [m.strip().lower() for m in methods_raw.replace(";", ",").split(",") if m.strip()]
        self._gradcam_methods = set(methods) if methods else set()
        if self._is_main_process():
            export_dir = getattr(config, "export_gradcam_dir", None)
            if export_dir:
                self._gradcam_dump_dir = export_dir
                os.makedirs(self._gradcam_dump_dir, exist_ok=True)
                for method in ("gnn", "llm"):
                    os.makedirs(os.path.join(self._gradcam_dump_dir, method), exist_ok=True)

        self._gnnexplainer_dump_dir = None
        self._gnnexplainer_warned_missing = False
        self._export_gnnexplainer_sids = list(getattr(config, "export_gnnexplainer_sids", []) or [])
        self._export_gnnexplainer_all = bool(getattr(config, "export_gnnexplainer_all", False))
        self._gnnexplainer_first_n = max(0, int(getattr(config, "export_gnnexplainer_first_n", 0)))
        self._gnnexplainer_export_count = {}
        self._gnnexplainer_steps = max(1, int(getattr(config, "gnnexplainer_steps", 5)))
        self._gnnexplainer_lr = float(getattr(config, "gnnexplainer_lr", 0.1))
        self._gnnexplainer_l1 = float(getattr(config, "gnnexplainer_l1", 0.05))
        self._gnnexplainer_export_cif_keep = bool(getattr(config, "export_gnnexplainer_cif_keep", False))
        self._gnnexplainer_export_scores = bool(getattr(config, "export_gnnexplainer_scores", False))
        if self._is_main_process():
            export_dir = getattr(config, "export_gnnexplainer_dir", None)
            if export_dir:
                self._gnnexplainer_dump_dir = export_dir
                os.makedirs(self._gnnexplainer_dump_dir, exist_ok=True)

    def logging(self, s, print_=True):
        if not self._is_main_process():
            return
        if print_:
            print(s)
        with open(self.log_path, "a") as f:
            f.write(s + "\n")

    # --------  pred  [B,1] --------
    def _reduce_pred_to_scalar(self, pred, mask, reduce="mean"):
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)  # [B] -> [B,1]
        if pred.dim() == 2 and pred.size(1) > 1:
            if mask is not None and mask.dim() == 2 and mask.size(0) == pred.size(0):
                m = mask.float()
                if reduce == "sum":
                    pred = (pred * m).sum(dim=1, keepdim=True)
                elif reduce == "mean":
                    denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
                    pred = (pred * m).sum(dim=1, keepdim=True) / denom
            else:
                if reduce == "sum":
                    pred = pred.sum(dim=1, keepdim=True)
                elif reduce == "mean":
                    pred = pred.mean(dim=1, keepdim=True)
        return pred

    # --------  --------
    def _per_atom_if_needed(self, pred_scalar, target_scalar, mask):
        natoms = mask.sum(dim=1, keepdim=True).float().clamp_min(1.0)
        if USE_PER_ATOM_LOSS:
            return pred_scalar / natoms, target_scalar / natoms, natoms
        else:
            return pred_scalar, target_scalar, natoms

    def _init_metric_tracker(self):
        return {"count": 0.0, "sum": 0.0, "sum_sq": 0.0, "sse": 0.0}

    def _update_metric_tracker(self, tracker, pred, target):
        diff = pred - target
        tracker["sse"] += torch.sum(diff ** 2).item()
        tracker["sum"] += torch.sum(target).item()
        tracker["sum_sq"] += torch.sum(target ** 2).item()
        tracker["count"] += float(target.numel())

    def _finalize_metrics(self, tracker):
        count = tracker["count"]
        if count <= 0:
            return {"rmse": float("nan"), "r2": float("nan")}
        mse = tracker["sse"] / count
        rmse = math.sqrt(mse)
        denom = tracker["sum_sq"] - (tracker["sum"] ** 2) / count
        if abs(denom) < 1e-12:
            r2 = float("nan")
        else:
            r2 = 1.0 - (tracker["sse"] / denom)
        return {"rmse": rmse, "r2": r2}

    # --------  --------
    def train(self, model, train_loader, val_loader, resume_state=None):
        base_model = model.module if hasattr(model, "module") else model
        base_model.enable_pruning = True
        base_model.force_pruning = True
        optimizer = base_model.configure_optimizers(
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )
        amp_enabled = bool(getattr(self.config, "amp", False))
        device_str = str(getattr(self.config, "device", ""))
        amp_enabled = amp_enabled and torch.cuda.is_available() and ("cuda" in device_str)
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        steps_per_epoch = len(train_loader)
        target_total_epochs = getattr(self.config, "target_total_epochs", self.config.epoch)
        epochs_to_run = max(1, getattr(self.config, "epochs_to_run", self.config.epoch))
        start_epoch = max(0, getattr(self.config, "resume_start_epoch", 0))
        total_steps = max(1, target_total_epochs * max(1, steps_per_epoch))
        explicit_warmup_steps = getattr(self.config, "phase_warmup_steps", None)
        explicit_warmup_ratio = getattr(self.config, "phase_warmup_ratio", None)
        if explicit_warmup_steps is not None:
            warmup_steps = max(1, int(explicit_warmup_steps))
        elif explicit_warmup_ratio is not None:
            warmup_steps = max(1, int(float(explicit_warmup_ratio) * total_steps))
        else:
            warmup_steps = max(1, int(0.05 * total_steps))

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(warmup_steps)
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        best_loss, best_epoch = 1e9, start_epoch
        best_ema_loss = 1e9
        early_stop_patience = int(getattr(self.config, "early_stop_patience", 0))
        early_stop_min_delta = float(getattr(self.config, "early_stop_min_delta", 0.0))
        epochs_no_improve = 0

        if resume_state:
            base_model.load_state_dict(resume_state.get('model_state_dict', {}), strict=False)
            opt_state = resume_state.get('optimizer_state_dict')
            if opt_state:
                try:
                    optimizer.load_state_dict(opt_state)
                except ValueError as exc:
                    self.logging(f"[Resume]  opt stateparam group mismatch{exc}")
                else:
                    device = getattr(base_model, "device", None)
                    if device is None:
                        device = next(base_model.parameters()).device
                    for state in optimizer.state.values():
                        for key, val in state.items():
                            if isinstance(val, torch.Tensor):
                                state[key] = val.to(device)
            best_loss = resume_state.get('loss', best_loss)
            best_epoch = resume_state.get('epoch', best_epoch)

        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch=start_epoch * max(1, steps_per_epoch) - 1
        )

        ema = None
        ema_decay = float(getattr(self.config, "ema_decay", 0.0))
        if ema_decay > 0.0:
            ema = ModelEMA(base_model, ema_decay)

        base_temperature = float(getattr(self.config, "temperature", 0.0))
        for local_epoch in range(epochs_to_run):
            epoch = start_epoch + local_epoch

            #  0
            current_temperature = base_temperature
            #  epoch epoch 
            base_model = model.module if hasattr(model, "module") else model
            setattr(base_model, "_current_epoch", epoch)
            self._reset_geom_logging(epoch)

            train_sampler = getattr(train_loader, "sampler", None)
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            # ----------  ----------
            model.train()
            train_loss, train_steps = 0.0, 0
            train_metric_tracker = self._init_metric_tracker()
            prefetcher = DataPreFetcher(train_loader)
            data = prefetcher.next()
            progress = None
            if self._is_main_process():
                progress = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{target_total_epochs}", leave=False)
            probe_limit = max(0, int(getattr(self.config, "probe_steps", 0)))
            log_interval = max(1, int(getattr(self.config, "period", 100)))
            while data is not None:
                if len(data) >= 12:
                    h, x, edges, target, raw_target, strings, mask, pos_mask, sid, sid_str, cells, pbcs = data
                elif len(data) >= 11:
                    h, x, edges, target, raw_target, strings, mask, pos_mask, sid, cells, pbcs = data
                    sid_str = None
                else:
                    h, x, edges, target, raw_target, strings, mask, pos_mask, sid = data
                    sid_str = None
                    cells, pbcs = None, None

                # EdgeDebug 

                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    outputs = model(
                        h, x, edges, strings, mask, target,
                        pos_mask=pos_mask,
                        temperature=current_temperature, train=True
                    )
                if isinstance(outputs, (list, tuple)):
                    pred = outputs[0]
                else:
                    pred = outputs
                if self._capture_geom and self._is_main_process():
                    geom_tensor = getattr(base_model, "last_geom_importance", None)
                    self._collect_geom_samples(epoch, sid, x, mask, pos_mask, geom_tensor)
                if getattr(self.config, "capture_text_attentions", False) and self._is_main_process():
                    batch_sids = sid[:, 0].detach().cpu().tolist()
                    global_step = epoch * max(1, steps_per_epoch) + train_steps
                    base_model.dump_text_attentions(batch_sids, phase="train", step=global_step)

                pred_scalar = self._reduce_pred_to_scalar(pred, mask, reduce="mean")
                tgt_scalar = raw_target

                pred_pa, tgt_pa, natoms = self._per_atom_if_needed(pred_scalar, tgt_scalar, mask)
                main_loss = self.train_loss(pred_pa, tgt_pa)

                total_loss = main_loss
                did_optimizer_step = False
                if amp_enabled:
                    scaler.scale(total_loss).backward()
                    clip_norm = float(getattr(self.config, "clip_grad_norm", 1.0))
                    if clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    prev_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    did_optimizer_step = scaler.get_scale() >= prev_scale
                else:
                    total_loss.backward()
                    clip_norm = float(getattr(self.config, "clip_grad_norm", 1.0))
                    if clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    optimizer.step()
                    did_optimizer_step = True
                if did_optimizer_step:
                    scheduler.step()
                    if ema is not None:
                        ema.update(base_model)

                train_loss += main_loss.item()
                train_steps += 1
                if self._is_main_process() and (train_steps % log_interval == 0):
                    lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
                    self.logging(
                        f"[TrainLoss] epoch={epoch:03d} step={train_steps} loss={main_loss.item():.6f} lr={lr:.3e}"
                    )
                with torch.no_grad():
                    self._update_metric_tracker(train_metric_tracker, pred_pa.detach(), tgt_pa.detach())

                data = prefetcher.next()
                if progress is not None:
                    progress.update(1)
                if self._is_main_process() and train_steps == 1:
                    raw_tokens = mask.float().sum(dim=1).mean().item()
                    total_len = mask.size(1)
                    keep_lengths = getattr(base_model.geom_token_pruner, "last_keep_lengths", None)
                    if keep_lengths is not None:
                        avg_keep = keep_lengths.float().mean().item()
                    else:
                        avg_keep = raw_tokens
                    self.logging(
                        f"[PruneStats] seq_len={total_len} raw_tokens={raw_tokens:.1f} kept_tokens={avg_keep:.1f}"
                    )
                    text_mask = None
                    if isinstance(strings, dict):
                        text_mask = strings.get("attention_mask", None)
                    else:
                        text_mask = getattr(strings, "attention_mask", None)
                    if text_mask is not None:
                        text_mask = text_mask.to(dtype=torch.float32)
                        text_raw = text_mask.sum(dim=1).mean().item()
                        text_len = text_mask.size(1)
                        self.logging(
                            f"[TextStats] seq_len={text_len} raw_tokens={text_raw:.1f} kept_tokens={text_raw:.1f}"
                        )
                    keep_mask = getattr(base_model.geom_token_pruner, "last_keep_mask", None)
                    if keep_mask is not None:
                        keep_mask = keep_mask.int()
                        zero_rows = (keep_mask.sum(dim=1) == keep_mask.size(1)).sum().item()
                        self.logging(f"[PruneDebug] keep_mask shape={keep_mask.shape}, rows_all_true={zero_rows}")
                    if keep_lengths is not None:
                        raw_vec = mask.float().sum(dim=1).cpu().tolist()
                        keep_vec = keep_lengths.float().cpu().tolist()
                        self.logging(f"[PruneDebug] raw_vec={raw_vec}")
                        self.logging(f"[PruneDebug] keep_vec={keep_vec}")
                    s_cls = getattr(base_model, "last_s_cls", None)
                    s_self = getattr(base_model, "last_s_self", None)
                    s_token = getattr(base_model, "last_s_token", None)
                    if s_cls is not None and s_self is not None and s_token is not None:
                        def _stat(t):
                            t = t.flatten().detach().cpu()
                            return t.mean().item(), t.min().item(), t.max().item()
                        cls_stat = _stat(s_cls)
                        self_stat = _stat(s_self)
                        token_stat = _stat(s_token)
                        self.logging(
                            f"[TIS Debug] S_cls mean/min/max={cls_stat[0]:.4f}/{cls_stat[1]:.4f}/{cls_stat[2]:.4f} | "
                            f"S_self mean/min/max={self_stat[0]:.4f}/{self_stat[1]:.4f}/{self_stat[2]:.4f} | "
                            f"S_token mean/min/max={token_stat[0]:.4f}/{token_stat[1]:.4f}/{token_stat[2]:.4f}"
                        )
                if probe_limit > 0 and train_steps >= probe_limit:
                    break

            if progress is not None:
                progress.close()
            avg_train = train_loss / max(1, train_steps)
            train_metrics = self._finalize_metrics(train_metric_tracker)
            self._flush_geom_samples(epoch)

            # ----------  ----------
            model.eval()
            if self._is_main_process():
                improved = False
                avg_val, val_metrics = self._run_validation(
                model, val_loader, current_temperature, 1.0, epoch
            )
                ema_val = None
                ema_metrics = None
                if ema is not None:
                    ema.store(base_model)
                    ema.copy_to(base_model)
                    ema_val, ema_metrics = self._run_validation(
                        model, val_loader, current_temperature, 1.0, epoch
                    )
                    ema.restore(base_model)

                phase_name = getattr(self.config, "current_phase", None)
                phase_prefix = f"[{phase_name}] " if phase_name else ""
                base_model._pre_keep_ratio = 1.0
                keep_lengths = getattr(base_model.geom_token_pruner, "last_keep_lengths", None)
                keep_info = ""
                if keep_lengths is not None:
                    keep_lengths = keep_lengths.float().cpu()
                    keep_info = f" | Avg keep tokens: {keep_lengths.mean().item():.1f}"
    
                extra = (
                    f" | Train RMSE: {train_metrics['rmse']:.4f} R2: {train_metrics['r2']:.4f}"
                    f" | Val RMSE: {val_metrics['rmse']:.4f} R2: {val_metrics['r2']:.4f}"
                    f"{keep_info}"
                )
                lr_val = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
                self.logging(
                    f"{phase_prefix}Epoch {epoch:03d} | Train MAE: {avg_train:.4f} | Val MAE: {avg_val:.4f}"
                    f" | LR: {lr_val:.2e}{extra}"
                )
                if ema_val is not None and ema_metrics is not None:
                    self.logging(
                        f"{phase_prefix}EMA Val MAE: {ema_val:.4f} | RMSE: {ema_metrics['rmse']:.4f} | R2: {ema_metrics['r2']:.4f}"
                    )
    
                if avg_val < (best_loss - early_stop_min_delta):
                    best_loss, best_epoch = avg_val, epoch
                    epochs_no_improve = 0
                    improved = True
                    best_state = {
                        'epoch': epoch,
                        'model_state_dict': base_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_val,
                    }
                    os.makedirs(self.save_dir, exist_ok=True)
                    torch.save(best_state, os.path.join(self.save_dir, 'best_model.pth'))
                    self.config.temperature = current_temperature
                    self.config.save_config(self.save_dir, filename="config_best.yaml")
                if ema_val is not None and ema_val < (best_ema_loss - early_stop_min_delta):
                    best_ema_loss = ema_val
                    best_state = {
                        'epoch': epoch,
                        'model_state_dict': ema.state_dict(),
                        'loss': ema_val,
                        'is_ema': True,
                    }
                    os.makedirs(self.save_dir, exist_ok=True)
                    torch.save(best_state, os.path.join(self.save_dir, 'best_ema_model.pth'))
                    self.config.temperature = current_temperature
                    self.config.save_config(self.save_dir, filename="config_best_ema.yaml")
                if not improved:
                    epochs_no_improve += 1
                    if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
                        self.logging(
                            f"[EarlyStop] no improvement for {epochs_no_improve} epochs "
                            f"(best epoch {best_epoch:03d}, best val {best_loss:.4f})."
                        )
                        break
            else:
                base_model._pre_keep_ratio = 1.0

        self.config.temperature = current_temperature
        if self._is_main_process():
            #  config_last.yaml
            self.config.save_config(self.save_dir, filename="config_last.yaml")

    def _reset_geom_logging(self, epoch):
        if self._is_main_process():
            self._cif_epoch_exported = set()
        if not self._capture_geom or not self._is_main_process():
            return
        self._geom_epoch_index = epoch
        self._geom_epoch_samples = []
        self._geom_epoch_flushed = False

    def _collect_geom_samples(self, epoch, sid, x, mask, pos_mask, geom_importance):
        if (
            (not self._capture_geom)
            or not self._is_main_process()
            or geom_importance is None
            or self._geom_dump_dir is None
        ):
            return
        if self._geom_epoch_index != epoch:
            self._reset_geom_logging(epoch)
        if len(self._geom_epoch_samples) >= self._geom_dump_count:
            return

        mask_cpu = mask.detach().to(device="cpu").bool()
        pos_mask_cpu = None
        if pos_mask is not None:
            pos_mask_cpu = pos_mask.detach().to(device="cpu").bool()
        coords = x.detach().to(device="cpu", dtype=torch.float32)[..., :3]
        geom_cpu = geom_importance.detach().to(device="cpu")
        sid_vals = None
        if sid is not None:
            sid_vals = sid.detach().to(device="cpu")

        batch_size = mask_cpu.size(0)
        for b in range(batch_size):
            if len(self._geom_epoch_samples) >= self._geom_dump_count:
                break
            valid_idx = torch.nonzero(mask_cpu[b], as_tuple=True)[0]
            if valid_idx.numel() == 0:
                continue
            ads_idx = None
            if pos_mask_cpu is not None:
                ads_idx = torch.nonzero((pos_mask_cpu[b] & mask_cpu[b]), as_tuple=True)[0]
            if ads_idx is None or ads_idx.numel() == 0:
                continue

            scores_tensor = geom_cpu[b, valid_idx]
            if scores_tensor.numel() == 0:
                continue
            coords_valid = coords[b, valid_idx]
            coords_ads = coords[b, ads_idx]
            dist_matrix = torch.cdist(
                coords_valid.unsqueeze(0),
                coords_ads.unsqueeze(0),
                p=2,
            ).squeeze(0)
            min_dist = dist_matrix.min(dim=1)[0]

            topk = min(5, scores_tensor.numel())
            top_vals, top_pos = torch.topk(scores_tensor, k=topk)
            top_nodes = valid_idx[top_pos]
            sid_value = None
            if sid_vals is not None:
                sid_flat = sid_vals[b].reshape(-1)
                if sid_flat.numel() > 0:
                    sid_value = float(sid_flat[0].item())
            record = {
                "epoch": int(epoch),
                "batch_index": int(b),
                "sid": sid_value,
                "node_indices": valid_idx.tolist(),
                "importance": scores_tensor.tolist(),
                "distances": min_dist.tolist(),
                "coords": coords_valid.tolist(),
                "ads_indices": ads_idx.tolist(),
                "ads_coords": coords_ads.tolist(),
                "top_nodes": top_nodes.tolist(),
                "top_scores": top_vals.tolist(),
                "top_distances": min_dist[top_pos].tolist(),
            }
            self._geom_epoch_samples.append(record)

    def _flush_geom_samples(self, epoch):
        if (
            (not self._capture_geom)
            or not self._is_main_process()
            or self._geom_dump_dir is None
            or self._geom_epoch_index != epoch
            or not self._geom_epoch_samples
        ):
            return
        out_path = os.path.join(self._geom_dump_dir, f"epoch_{epoch+1:03d}.json")
        payload = {"epoch": int(epoch), "samples": self._geom_epoch_samples}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self._geom_epoch_flushed = True
        #  reset 
    def _log_stoken_sort(self, model, mask, sid, tag, batch_idx=0, topk=10):
        if not self._is_main_process():
            return
        base_model = model.module if hasattr(model, "module") else model
        s_token = getattr(base_model, "last_s_token_pre", None)
        dists = getattr(base_model, "last_token_dist", None)
        if s_token is None:
            return
        mask_bool = mask.detach().to(device="cpu").bool()
        s_cpu = s_token.detach().to(device="cpu")
        if s_cpu.size(0) == 0:
            return
        b = 0  # 
        valid_idx = torch.nonzero(mask_bool[b], as_tuple=True)[0]
        if valid_idx.numel() == 0:
            return
        s_vals = s_cpu[b, valid_idx]
        dist_vals = None
        if dists and len(dists) > b and dists[b] is not None:
            dist_vals = dists[b]
            if hasattr(dist_vals, "tolist"):
                dist_vals = dist_vals.tolist()
        entries = []
        for j, idx_val in enumerate(valid_idx.tolist()):
            dist_j = dist_vals[j] if dist_vals is not None and j < len(dist_vals) else None
            entries.append((idx_val, float(s_vals[j].item()), dist_j))
        entries_sorted = sorted(entries, key=lambda x: x[1], reverse=True)
        if topk is not None and topk > 0:
            entries_sorted = entries_sorted[:topk]
        sid_val = None
        if sid is not None:
            if sid.dim() == 2:
                sid_val = int(sid[b, 0].item())
            else:
                sid_val = int(sid[b].item())
        self.logging(f"[S_TOKEN_SORT] {tag} batch={batch_idx} sid={sid_val} top{len(entries_sorted)}={entries_sorted}")

    def _run_validation(
        self,
        model,
        val_loader,
        current_temperature,
        subset_ratio,
        epoch_idx=0,
        desc=None,
        pruning_mode=None,
    ):
        total_val_batches = len(val_loader)
        if total_val_batches == 0:
            return float("nan"), {"rmse": float("nan"), "r2": float("nan")}
        base_model = model.module if hasattr(model, "module") else model
        if isinstance(pruning_mode, str):
            pruning_mode = pruning_mode.strip().lower()
        save_preds = bool(getattr(self.config, "save_val_preds", False)) and self._is_main_process()
        pred_list = []
        tgt_list = []
        sid_list = []
        max_val_steps = total_val_batches
        if subset_ratio is None or subset_ratio <= 0:
            max_val_steps = total_val_batches
        elif subset_ratio < 1.0:
            max_val_steps = max(1, math.ceil(total_val_batches * subset_ratio))
        prefetcher = DataPreFetcher(val_loader)
        data = prefetcher.next()
        val_loss, val_steps = 0.0, 0
        metric_tracker = self._init_metric_tracker()
        progress = None
        if self._is_main_process():
            progress_desc = desc or f"Val {epoch_idx+1}"
            progress = tqdm(total=max_val_steps, desc=progress_desc, leave=False)
        with torch.no_grad():
            while data is not None and (max_val_steps <= 0 or val_steps < max_val_steps):
                if len(data) >= 12:
                    h, x, edges, target, raw_target, strings, mask, pos_mask, sid, sid_str, cells, pbcs = data
                elif len(data) >= 11:
                    h, x, edges, target, raw_target, strings, mask, pos_mask, sid, cells, pbcs = data
                    sid_str = None
                else:
                    h, x, edges, target, raw_target, strings, mask, pos_mask, sid = data
                    sid_str = None
                    cells, pbcs = None, None
                base_model.enable_pruning = True
                base_model.force_pruning = True
                outputs = model(
                    h, x, edges, strings, mask, target,
                    pos_mask=pos_mask,
                    temperature=current_temperature, train=False
                )
                if isinstance(outputs, (list, tuple)):
                    pred = outputs[0]
                else:
                    pred = outputs
                if getattr(self.config, "capture_text_attentions", False) and self._is_main_process():
                    base_model = model.module if hasattr(model, "module") else model
                    batch_sids = sid[:, 0].detach().cpu().tolist()
                    span = max(1, max_val_steps)
                    step_id = epoch_idx * span + val_steps
                    base_model.dump_text_attentions(batch_sids, phase="val", step=step_id)
                pred_scalar = self._reduce_pred_to_scalar(pred, mask, reduce="mean")
                tgt_scalar = raw_target
                pred_pa, tgt_pa, natoms = self._per_atom_if_needed(pred_scalar, tgt_scalar, mask)
                self._update_metric_tracker(metric_tracker, pred_pa, tgt_pa)
                val_loss += self.train_loss(pred_pa, tgt_pa).item()
                if save_preds:
                    pred_list.append(pred_pa.detach().cpu().reshape(-1).numpy())
                    tgt_list.append(tgt_pa.detach().cpu().reshape(-1).numpy())
                    if sid is not None:
                        if torch.is_tensor(sid):
                            if sid.dim() == 2:
                                sid_vals = sid[:, 0].detach().cpu().tolist()
                            else:
                                sid_vals = sid.detach().cpu().tolist()
                        else:
                            try:
                                sid_vals = list(sid)
                            except Exception:
                                sid_vals = [str(sid)]
                        sid_list.extend(sid_vals)
                val_steps += 1
                if self._is_main_process() and getattr(self.config, "mode", "train") == "val":
                    self._maybe_export_pruned_cif(
                        base_model, epoch_idx, "val", h, x, edges, mask, sid, cells, pbcs,
                        strings=strings, target=target, pos_mask=pos_mask, temperature=current_temperature,
                        sid_str=sid_str,
                    )
                    self._maybe_export_gradcam(
                        model, base_model, epoch_idx, "val", h, x, edges, mask, pos_mask, sid,
                        strings=strings, temperature=current_temperature, sid_str=sid_str,
                        cells=cells, pbcs=pbcs,
                    )
                    self._maybe_export_gnnexplainer(
                        base_model, epoch_idx, "val", h, x, edges, mask, pos_mask, sid,
                        strings=strings, temperature=current_temperature, sid_str=sid_str,
                        cells=cells, pbcs=pbcs,
                    )
                if progress is not None:
                    progress.update(1)
                if max_val_steps > 0 and val_steps >= max_val_steps:
                    break
                data = prefetcher.next()
        if progress is not None:
            progress.close()
        if save_preds and pred_list:
            import numpy as np
            preds_all = np.concatenate(pred_list, axis=0)
            tgts_all = np.concatenate(tgt_list, axis=0)
            if sid_list:
                sids_all = np.array(sid_list, dtype=object)
            else:
                sids_all = np.array([], dtype=object)
            save_path = getattr(self.config, "save_val_preds_path", "") or ""
            if not save_path:
                save_path = os.path.join(self.log_dir, f"val_preds_epoch_{epoch_idx+1:03d}.npz")
            np.savez(save_path, pred=preds_all, target=tgts_all, sid=sids_all)
        avg_val = val_loss / max(1, val_steps)
        return avg_val, self._finalize_metrics(metric_tracker)

    def _maybe_export_pruned_cif(
        self,
        base_model,
        epoch_idx,
        phase,
        h,
        x,
        edges,
        mask,
        sid,
        cells,
        pbcs,
        strings=None,
        target=None,
        pos_mask=None,
        temperature=0.0,
        sid_str=None,
    ):
        if (not self._export_pruned_cif_sids and not bool(getattr(self.config, "export_pruned_cif_all", False))) or self._cif_dump_dir is None:
            return
        if self._export_pruned_cif_mode == "off":
            return
        if self._export_pruned_cif_mode not in {"both", phase}:
            return
        if cells is None or pbcs is None:
            if not self._cif_warned_missing:
                self.logging("[CIF Export] Missing cell/pbc in batch; skip CIF export.")
                self._cif_warned_missing = True
            return
        keep_mask_full = getattr(base_model, "last_geom_keep_mask", None)
        if keep_mask_full is None and strings is not None:
            base_model.enable_pruning = True
            base_model.force_pruning = True
            _ = base_model(
                h, x, edges=edges, strings=strings, mask=mask, target=target,
                pos_mask=pos_mask, temperature=temperature, train=False
            )
            keep_mask_full = getattr(base_model, "last_geom_keep_mask", None)
        if keep_mask_full is None:
            if not self._cif_warned_missing:
                self.logging("[CIF Export] Pruning not enabled or keep_mask missing; skip CIF export.")
                self._cif_warned_missing = True
            return

        batch_size = h.size(0)
        for b in range(batch_size):
            sid_val = None
            try:
                if sid.dim() == 2:
                    sid_val = int(sid[b, 0].item())
                else:
                    sid_val = int(sid[b].item())
            except Exception:
                continue
            if not bool(getattr(self.config, "export_pruned_cif_all", False)):
                if sid_val not in self._export_pruned_cif_sids:
                    continue
            key = (phase, int(epoch_idx), sid_val)
            if key in self._cif_epoch_exported:
                continue
            self._cif_epoch_exported.add(key)
            cell = cells[b]
            pbc = pbcs[b]
            tis_b = getattr(base_model, "last_tis_pre", None)
            sid_label = None
            if sid_str is not None:
                try:
                    sid_label = sid_str[b]
                except Exception:
                    sid_label = None
            self._write_cif_pair(
                sid_val,
                int(epoch_idx),
                phase,
                h[b],
                x[b],
                mask[b],
                keep_mask_full[b],
                cell,
                pbc,
                sid_label=sid_label,
                tis_b=tis_b[b] if tis_b is not None and hasattr(tis_b, "dim") and tis_b.dim() > 1 else tis_b,
            )

    def _maybe_export_gradcam(
        self,
        model,
        base_model,
        epoch_idx,
        phase,
        h,
        x,
        edges,
        mask,
        pos_mask,
        sid,
        strings,
        temperature=0.0,
        sid_str=None,
        cells=None,
        pbcs=None,
    ):
        if self._gradcam_dump_dir is None or not self._gradcam_methods:
            return
        if (not self._export_gradcam_all) and not self._export_gradcam_sids:
            return
        if mask is None or strings is None:
            if not self._gradcam_warned_missing:
                self.logging("[GradCAM Export] Missing mask/strings in batch; skip.")
                self._gradcam_warned_missing = True
            return
        if self._gradcam_export_cif_keep and (cells is None or pbcs is None):
            if not self._gradcam_warned_missing:
                self.logging("[GradCAM Export] Missing cell/pbc in batch; skip CIF export.")
                self._gradcam_warned_missing = True
            return

        keep_mask_full = getattr(base_model, "last_geom_keep_mask", None)
        if keep_mask_full is None:
            base_model.enable_pruning = True
            base_model.force_pruning = True
            _ = base_model(
                h, x, edges=edges, strings=strings, mask=mask, target=None,
                pos_mask=pos_mask, temperature=temperature, train=False
            )
            keep_mask_full = getattr(base_model, "last_geom_keep_mask", None)

        prev_capture_gnn = getattr(base_model, "_capture_gradcam_gnn", False)
        prev_capture_llm = getattr(base_model, "_capture_gradcam_llm", False)
        base_model._capture_gradcam_gnn = "gnn" in self._gradcam_methods
        base_model._capture_gradcam_llm = "llm" in self._gradcam_methods

        base_model.enable_pruning = True
        base_model.force_pruning = True

        with torch.enable_grad():
            if hasattr(model, "zero_grad"):
                model.zero_grad(set_to_none=True)
            outputs = model(
                h, x, edges, strings, mask, target=None,
                pos_mask=pos_mask, temperature=temperature, train=False
            )
            pred = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            pred_vec = pred.view(-1)
            act_gnn = getattr(base_model, "last_gradcam_gnn_act", None)
            act_text = getattr(base_model, "last_gradcam_text_act", None)
            text2geom_attn = getattr(base_model, "last_text2geom_attn", None)
            text_mask = getattr(base_model, "last_text_mask", None)

            mask_bool = mask.to(dtype=torch.bool)
            pos_mask_bool = pos_mask.to(dtype=torch.bool) if pos_mask is not None else None
            min_non_ads = getattr(self.config, "gradcam_min_non_ads", None)
            max_non_ads = getattr(self.config, "gradcam_max_non_ads", None)
            if min_non_ads is None:
                min_non_ads = getattr(self.config, "geom_prune_min_non_ads", 0) or 0
            if max_non_ads is None:
                max_non_ads = getattr(self.config, "geom_prune_max_non_ads", None)

            batch_size = pred_vec.size(0)
            for b in range(batch_size):
                sid_val = None
                try:
                    if sid.dim() == 2:
                        sid_val = int(sid[b, 0].item())
                    else:
                        sid_val = int(sid[b].item())
                except Exception:
                    continue
                if not self._export_gradcam_all and sid_val not in self._export_gradcam_sids:
                    continue
                key = (phase, int(epoch_idx), sid_val)
                if key in self._gradcam_epoch_exported:
                    continue
                count_key = (phase, int(epoch_idx))
                if self._gradcam_first_n > 0:
                    cur_count = int(self._gradcam_export_count.get(count_key, 0))
                    if cur_count >= self._gradcam_first_n:
                        continue

                sid_label = None
                if sid_str is not None:
                    try:
                        sid_label = sid_str[b]
                    except Exception:
                        sid_label = None

                ads_idx = []
                if pos_mask_bool is not None:
                    ads_idx = torch.nonzero(pos_mask_bool[b] & mask_bool[b], as_tuple=True)[0].tolist()
                if pos_mask_bool is not None:
                    non_ads_idx = torch.nonzero(mask_bool[b] & (~pos_mask_bool[b]), as_tuple=True)[0]
                else:
                    non_ads_idx = torch.nonzero(mask_bool[b], as_tuple=True)[0]
                if non_ads_idx.numel() == 0:
                    keep_idx = sorted(ads_idx)
                    keep_idx_full = self._map_keep_indices(mask_bool[b], keep_idx)
                    self._write_gradcam_keep_idx(
                        sid_val, epoch_idx, phase, keep_idx_full, sid_label=sid_label, method="empty"
                    )
                    if self._gradcam_export_cif_keep:
                        self._write_gradcam_keep_cif(
                            sid_val, epoch_idx, phase, h[b], x[b], mask[b], keep_idx_full,
                            cells[b], pbcs[b], sid_label=sid_label, method="empty"
                        )
                    self._gradcam_export_count[count_key] = int(self._gradcam_export_count.get(count_key, 0)) + 1
                    self._gradcam_epoch_exported.add(key)
                    continue

                k_non_ads = None
                if use_pruned_k and keep_mask_full is not None:
                    keep_b = keep_mask_full[b].to(device=mask_bool.device, dtype=torch.bool)
                    if pos_mask_bool is not None:
                        k_non_ads = int((keep_b & (~pos_mask_bool[b])).sum().item())
                    else:
                        k_non_ads = int(keep_b.sum().item())
                if k_non_ads is None or k_non_ads <= 0:
                    k_non_ads = int(non_ads_idx.numel())
                if max_non_ads is not None:
                    k_non_ads = min(k_non_ads, int(max_non_ads))
                if min_non_ads is not None and k_non_ads < int(min_non_ads):
                    k_non_ads = min(int(min_non_ads), int(non_ads_idx.numel()))
                k_non_ads = max(0, min(int(non_ads_idx.numel()), k_non_ads))

                if "gnn" in self._gradcam_methods and act_gnn is not None:
                    grad_gnn = torch.autograd.grad(
                        pred_vec[b], act_gnn, retain_graph=True, allow_unused=True
                    )[0]
                    if grad_gnn is not None:
                        scores = torch.relu((grad_gnn[b] * act_gnn[b]).sum(dim=-1))
                        scores_non = scores[non_ads_idx]
                        topk = min(k_non_ads, scores_non.numel())
                        _, top_pos = torch.topk(scores_non, k=topk) if topk > 0 else (None, None)
                        keep_non = non_ads_idx[top_pos].tolist() if topk > 0 else []
                        keep_idx = sorted(set(ads_idx + keep_non))
                        keep_idx_full = self._map_keep_indices(mask_bool[b], keep_idx)
                        self._write_gradcam_keep_idx(
                            sid_val, epoch_idx, phase, keep_idx_full, sid_label=sid_label, method="gnn"
                        )
                        if self._gradcam_export_cif_keep:
                            self._write_gradcam_keep_cif(
                                sid_val, epoch_idx, phase, h[b], x[b], mask[b], keep_idx_full,
                                cells[b], pbcs[b], sid_label=sid_label, method="gnn"
                            )
                        if self._gradcam_export_scores:
                            scores_full = self._map_scores(mask_bool[b], scores)
                            self._write_gradcam_scores(
                                sid_val, epoch_idx, phase, scores_full, sid_label=sid_label, method="gnn"
                            )

                if "llm" in self._gradcam_methods and act_text is not None and text2geom_attn is not None:
                    grad_llm = torch.autograd.grad(
                        pred_vec[b], act_text, retain_graph=True, allow_unused=True
                    )[0]
                    if grad_llm is not None:
                        token_scores = torch.relu((grad_llm[b] * act_text[b]).sum(dim=-1))
                        if text_mask is not None:
                            token_scores = token_scores * text_mask[b].to(dtype=token_scores.dtype)
                        attn_b = text2geom_attn[b]  # [T, G]
                        atom_scores = torch.matmul(attn_b.transpose(0, 1), token_scores)
                        scores_non = atom_scores[non_ads_idx]
                        topk = min(k_non_ads, scores_non.numel())
                        _, top_pos = torch.topk(scores_non, k=topk) if topk > 0 else (None, None)
                        keep_non = non_ads_idx[top_pos].tolist() if topk > 0 else []
                        keep_idx = sorted(set(ads_idx + keep_non))
                        keep_idx_full = self._map_keep_indices(mask_bool[b], keep_idx)
                        self._write_gradcam_keep_idx(
                            sid_val, epoch_idx, phase, keep_idx_full, sid_label=sid_label, method="llm"
                        )
                        if self._gradcam_export_cif_keep:
                            self._write_gradcam_keep_cif(
                                sid_val, epoch_idx, phase, h[b], x[b], mask[b], keep_idx_full,
                                cells[b], pbcs[b], sid_label=sid_label, method="llm"
                            )
                        if self._gradcam_export_scores:
                            scores_full = self._map_scores(mask_bool[b], atom_scores)
                            self._write_gradcam_scores(
                                sid_val, epoch_idx, phase, scores_full, sid_label=sid_label, method="llm"
                            )

                self._gradcam_export_count[count_key] = int(self._gradcam_export_count.get(count_key, 0)) + 1
                self._gradcam_epoch_exported.add(key)

        base_model._capture_gradcam_gnn = prev_capture_gnn
        base_model._capture_gradcam_llm = prev_capture_llm

    def _write_gradcam_keep_idx(self, sid_val, epoch_idx, phase, keep_idx, sid_label=None, method="gnn"):
        if self._gradcam_dump_dir is None:
            return
        base = f"sid_{sid_val}"
        if sid_label:
            base = f"{base}_{sid_label}"
        fname = f"{base}_epoch_{epoch_idx+1:03d}_{phase}_gradcam_{method}_keep_idx.txt"
        out_dir = os.path.join(self._gradcam_dump_dir, method)
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(",".join(str(i) for i in keep_idx))

    def _map_keep_indices(self, mask_bool, keep_idx):
        valid_idx = torch.nonzero(mask_bool, as_tuple=True)[0]
        if valid_idx.numel() == 0:
            return []
        keep_set = set(int(i) for i in keep_idx)
        keep_full = []
        for j, idx in enumerate(valid_idx.tolist()):
            if idx in keep_set:
                keep_full.append(j)
        return keep_full

    def _map_scores(self, mask_bool, scores):
        valid_idx = torch.nonzero(mask_bool, as_tuple=True)[0]
        if valid_idx.numel() == 0:
            return []
        scores_valid = scores[valid_idx]
        return scores_valid.detach().cpu().tolist()

    def _write_gradcam_scores(self, sid_val, epoch_idx, phase, scores_full, sid_label=None, method="gnn"):
        if self._gradcam_dump_dir is None:
            return
        base = f"sid_{sid_val}"
        if sid_label:
            base = f"{base}_{sid_label}"
        fname = f"{base}_epoch_{epoch_idx+1:03d}_{phase}_gradcam_{method}_scores.txt"
        out_dir = os.path.join(self._gradcam_dump_dir, method)
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("index,score\n")
            for idx, val in enumerate(scores_full):
                f.write(f"{idx},{val}\n")

    def _write_gradcam_keep_cif(
        self,
        sid_val,
        epoch_idx,
        phase,
        h_b,
        x_b,
        mask_b,
        keep_idx_full,
        cell,
        pbc,
        sid_label=None,
        method="gnn",
    ):
        if self._gradcam_dump_dir is None:
            return
        try:
            import numpy as np
            from ase import Atoms
            from ase.io import write
        except Exception as exc:
            if not self._gradcam_warned_missing:
                self.logging(f"[GradCAM Export] ASE not available: {exc}")
                self._gradcam_warned_missing = True
            return

        mask_bool = mask_b.detach().cpu().bool().numpy()
        h_cpu = h_b.detach().cpu().numpy()
        x_cpu = x_b.detach().cpu().numpy()
        numbers_full = h_cpu[mask_bool].tolist()
        coords_full = x_cpu[mask_bool]
        if not keep_idx_full:
            return
        keep_mask = np.zeros(len(numbers_full), dtype=bool)
        for idx in keep_idx_full:
            if 0 <= idx < len(keep_mask):
                keep_mask[idx] = True
        numbers_keep = [num for num, keep in zip(numbers_full, keep_mask) if keep]
        coords_keep = coords_full[keep_mask]
        if len(numbers_keep) == 0:
            return

        cell_np = np.asarray(cell, dtype=float)
        pbc_np = np.asarray(pbc, dtype=bool)
        if sid_label is None:
            base_id = f"sid_{sid_val}"
        else:
            safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(sid_label))
            base_id = f"sid_{sid_val}_{safe}"
        base_name = f"{base_id}_epoch_{epoch_idx+1:03d}_{phase}_gradcam_{method}"
        out_dir = os.path.join(self._gradcam_dump_dir, method)
        out_path = os.path.join(out_dir, f"{base_name}_keep.cif")
        atoms_keep = Atoms(numbers=numbers_keep, positions=coords_keep, cell=cell_np, pbc=pbc_np)
        write(out_path, atoms_keep, format="cif")

    def _write_gnnexplainer_keep_cif(
        self,
        sid_val,
        epoch_idx,
        phase,
        h_b,
        x_b,
        mask_b,
        keep_idx_full,
        cell,
        pbc,
        sid_label=None,
    ):
        if self._gnnexplainer_dump_dir is None:
            return
        try:
            import numpy as np
            from ase import Atoms
            from ase.io import write
        except Exception as exc:
            if not self._gnnexplainer_warned_missing:
                self.logging(f"[GNNExplainer Export] ASE not available: {exc}")
                self._gnnexplainer_warned_missing = True
            return

        mask_bool = mask_b.detach().cpu().bool().numpy()
        h_cpu = h_b.detach().cpu().numpy()
        x_cpu = x_b.detach().cpu().numpy()
        numbers_full = h_cpu[mask_bool].tolist()
        coords_full = x_cpu[mask_bool]
        if not keep_idx_full:
            return
        keep_mask = np.zeros(len(numbers_full), dtype=bool)
        for idx in keep_idx_full:
            if 0 <= idx < len(keep_mask):
                keep_mask[idx] = True
        numbers_keep = [num for num, keep in zip(numbers_full, keep_mask) if keep]
        coords_keep = coords_full[keep_mask]
        if len(numbers_keep) == 0:
            return

        cell_np = np.asarray(cell, dtype=float)
        pbc_np = np.asarray(pbc, dtype=bool)
        if sid_label is None:
            base_id = f"sid_{sid_val}"
        else:
            safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(sid_label))
            base_id = f"sid_{sid_val}_{safe}"
        base_name = f"{base_id}_epoch_{epoch_idx+1:03d}_{phase}_gnnexplainer"
        out_path = os.path.join(self._gnnexplainer_dump_dir, f"{base_name}_keep.cif")
        atoms_keep = Atoms(numbers=numbers_keep, positions=coords_keep, cell=cell_np, pbc=pbc_np)
        write(out_path, atoms_keep, format="cif")

    def _slice_text_batch(self, strings, b):
        if strings is None:
            return None
        if isinstance(strings, dict):
            return {k: v[b:b+1] for k, v in strings.items()}
        if hasattr(strings, "data") and isinstance(strings.data, dict):
            return {k: v[b:b+1] for k, v in strings.data.items()}
        try:
            return {k: v[b:b+1] for k, v in strings.items()}
        except Exception:
            return strings

    def _maybe_export_gnnexplainer(
        self,
        base_model,
        epoch_idx,
        phase,
        h,
        x,
        edges,
        mask,
        pos_mask,
        sid,
        strings,
        temperature=0.0,
        sid_str=None,
        cells=None,
        pbcs=None,
    ):
        if self._gnnexplainer_dump_dir is None:
            return
        if (not self._export_gnnexplainer_all) and not self._export_gnnexplainer_sids:
            return
        if mask is None or strings is None:
            if not self._gnnexplainer_warned_missing:
                self.logging("[GNNExplainer Export] Missing mask/strings in batch; skip.")
                self._gnnexplainer_warned_missing = True
            return
        if self._gnnexplainer_export_cif_keep and (cells is None or pbcs is None):
            if not self._gnnexplainer_warned_missing:
                self.logging("[GNNExplainer Export] Missing cell/pbc in batch; skip CIF export.")
                self._gnnexplainer_warned_missing = True
            return

        mask_bool = mask.to(dtype=torch.bool)
        pos_mask_bool = pos_mask.to(dtype=torch.bool) if pos_mask is not None else None
        keep_mask_full = getattr(base_model, "last_geom_keep_mask", None)
        if keep_mask_full is None:
            base_model.enable_pruning = True
            base_model.force_pruning = True
            _ = base_model(
                h, x, edges=edges, strings=strings, mask=mask, target=None,
                pos_mask=pos_mask, temperature=temperature, train=False
            )
            keep_mask_full = getattr(base_model, "last_geom_keep_mask", None)

        min_non_ads = getattr(self.config, "gradcam_min_non_ads", None)
        max_non_ads = getattr(self.config, "gradcam_max_non_ads", None)
        if min_non_ads is None:
            min_non_ads = getattr(self.config, "geom_prune_min_non_ads", 0) or 0
        if max_non_ads is None:
            max_non_ads = getattr(self.config, "geom_prune_max_non_ads", None)

        prev_explainer = getattr(base_model, "enable_explainer_mask", False)
        prev_mask = getattr(base_model, "explainer_mask", None)
        base_model.enable_pruning = True
        base_model.force_pruning = True
        base_model.enable_explainer_mask = True

        with torch.enable_grad():
            batch_size = h.size(0)
            for b in range(batch_size):
                sid_val = None
                try:
                    if sid.dim() == 2:
                        sid_val = int(sid[b, 0].item())
                    else:
                        sid_val = int(sid[b].item())
                except Exception:
                    continue
                if not self._export_gnnexplainer_all and sid_val not in self._export_gnnexplainer_sids:
                    continue
                count_key = (phase, int(epoch_idx))
                if self._gnnexplainer_first_n > 0:
                    cur_count = int(self._gnnexplainer_export_count.get(count_key, 0))
                    if cur_count >= self._gnnexplainer_first_n:
                        continue

                sid_label = None
                if sid_str is not None:
                    try:
                        sid_label = sid_str[b]
                    except Exception:
                        sid_label = None

                mask_b = mask_bool[b]
                pos_b = pos_mask_bool[b] if pos_mask_bool is not None else None
                valid_idx = torch.nonzero(mask_b, as_tuple=True)[0]
                if valid_idx.numel() == 0:
                    continue
                if pos_b is not None:
                    ads_idx = torch.nonzero(pos_b & mask_b, as_tuple=True)[0]
                    non_ads_idx = torch.nonzero(mask_b & (~pos_b), as_tuple=True)[0]
                else:
                    ads_idx = torch.empty(0, device=mask_b.device, dtype=torch.long)
                    non_ads_idx = valid_idx
                if non_ads_idx.numel() == 0:
                    continue

                strings_b = self._slice_text_batch(strings, b)

                with torch.no_grad():
                    base_model.explainer_mask = None
                    pred0 = base_model(
                        h[b:b+1], x[b:b+1], edges=edges[b:b+1], strings=strings_b,
                        mask=mask[b:b+1], target=None, pos_mask=pos_mask[b:b+1] if pos_mask is not None else None,
                        temperature=temperature, train=False
                    )
                    pred0 = pred0[0] if isinstance(pred0, (list, tuple)) else pred0

                param = torch.zeros(non_ads_idx.numel(), device=h.device, requires_grad=True)
                optimizer = torch.optim.Adam([param], lr=self._gnnexplainer_lr)
                for _ in range(self._gnnexplainer_steps):
                    if hasattr(base_model, "zero_grad"):
                        base_model.zero_grad(set_to_none=True)
                    optimizer.zero_grad(set_to_none=True)
                    mask_vals = torch.sigmoid(param)
                    full_mask = torch.zeros(mask_b.size(0), device=h.device)
                    full_mask[valid_idx] = 1.0
                    full_mask[non_ads_idx] = mask_vals
                    if ads_idx.numel() > 0:
                        full_mask[ads_idx] = 1.0
                    base_model.explainer_mask = full_mask.unsqueeze(0)
                    pred1 = base_model(
                        h[b:b+1], x[b:b+1], edges=edges[b:b+1], strings=strings_b,
                        mask=mask[b:b+1], target=None, pos_mask=pos_mask[b:b+1] if pos_mask is not None else None,
                        temperature=temperature, train=False
                    )
                    pred1 = pred1[0] if isinstance(pred1, (list, tuple)) else pred1
                    loss = (pred1 - pred0.detach()) ** 2
                    loss = loss.mean() + self._gnnexplainer_l1 * mask_vals.mean()
                    loss.backward()
                    optimizer.step()

                mask_vals = torch.sigmoid(param).detach()
                scores_full = torch.zeros(mask_b.size(0), device=h.device)
                scores_full[non_ads_idx] = mask_vals
                if ads_idx.numel() > 0:
                    scores_full[ads_idx] = 1.0
                scores_full = scores_full.detach().cpu().tolist()
                keep_k = None
                if use_pruned_k and keep_mask_full is not None:
                    keep_b = keep_mask_full[b].to(device=mask_b.device, dtype=torch.bool)
                    if pos_b is not None:
                        keep_k = int((keep_b & (~pos_b)).sum().item())
                    else:
                        keep_k = int(keep_b.sum().item())
                if keep_k is None or keep_k <= 0:
                    keep_k = int(non_ads_idx.numel())
                if max_non_ads is not None:
                    keep_k = min(keep_k, int(max_non_ads))
                if min_non_ads is not None and keep_k < int(min_non_ads):
                    keep_k = min(int(min_non_ads), int(non_ads_idx.numel()))

                scores_non = torch.tensor(scores_full, device=h.device)[non_ads_idx]
                topk = min(keep_k, scores_non.numel())
                _, top_pos = torch.topk(scores_non, k=topk) if topk > 0 else (None, None)
                keep_non = non_ads_idx[top_pos].tolist() if topk > 0 else []
                keep_idx = sorted(set(ads_idx.tolist() + keep_non))
                keep_idx_full = self._map_keep_indices(mask_b, keep_idx)

                base = f"sid_{sid_val}"
                if sid_label:
                    base = f"{base}_{sid_label}"
                name_base = f"{base}_epoch_{epoch_idx+1:03d}_{phase}_gnnexplainer"
                keep_path = os.path.join(self._gnnexplainer_dump_dir, f"{name_base}_keep_idx.txt")
                with open(keep_path, "w", encoding="utf-8") as f:
                    f.write(",".join(str(i) for i in keep_idx_full))
                if self._gnnexplainer_export_scores:
                    scores_path = os.path.join(self._gnnexplainer_dump_dir, f"{name_base}_scores.txt")
                    with open(scores_path, "w", encoding="utf-8") as f:
                        f.write("index,score\n")
                        valid_list = torch.nonzero(mask_b, as_tuple=True)[0].tolist()
                        for idx, val in enumerate([scores_full[i] for i in valid_list]):
                            f.write(f"{idx},{val}\n")
                if self._gnnexplainer_export_cif_keep:
                    self._write_gnnexplainer_keep_cif(
                        sid_val, epoch_idx, phase, h[b], x[b], mask[b], keep_idx_full,
                        cells[b], pbcs[b], sid_label=sid_label
                    )

                self._gnnexplainer_export_count[count_key] = int(self._gnnexplainer_export_count.get(count_key, 0)) + 1

        base_model.enable_explainer_mask = prev_explainer
        base_model.explainer_mask = prev_mask

    def _write_cif_pair(self, sid_val, epoch_idx, phase, h_b, x_b, mask_b, keep_mask_b, cell, pbc, sid_label=None, tis_b=None):
        try:
            import numpy as np
            from ase import Atoms
            from ase.io import write
            from ase.data import chemical_symbols
        except Exception as exc:
            if not self._cif_warned_missing:
                self.logging(f"[CIF Export] ASE not available: {exc}")
                self._cif_warned_missing = True
            return

        mask_bool = mask_b.detach().cpu().bool().numpy()
        keep_mask = keep_mask_b.detach().cpu().bool().numpy()
        keep_mask = keep_mask & mask_bool
        if not keep_mask.any():
            return
        h_cpu = h_b.detach().cpu().numpy()
        x_cpu = x_b.detach().cpu().numpy()
        numbers_full = h_cpu[mask_bool].tolist()
        coords_full = x_cpu[mask_bool]
        keep_mask_full = keep_mask[mask_bool]
        numbers_keep = h_cpu[keep_mask].tolist()
        coords_keep = x_cpu[keep_mask]

        cell_np = np.asarray(cell, dtype=float)
        pbc_np = np.asarray(pbc, dtype=bool)
        if sid_label is None:
            base_id = f"sid_{sid_val}"
        else:
            safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(sid_label))
            base_id = f"sid_{sid_val}_{safe}"
        base_name = f"{base_id}_epoch_{epoch_idx+1:03d}_{phase}"
        full_path = os.path.join(self._cif_dump_dir, f"{base_name}_full.cif")
        pruned_path = os.path.join(self._cif_dump_dir, f"{base_name}_pruned.cif")
        keep_idx_path = os.path.join(self._cif_dump_dir, f"{base_name}_keep_idx.txt")
        tis_path = os.path.join(self._cif_dump_dir, f"{base_name}_tis.txt")

        atoms_full = Atoms(numbers=numbers_full, positions=coords_full, cell=cell_np, pbc=pbc_np)
        write(full_path, atoms_full, format="cif")
        atoms_pruned = Atoms(numbers=numbers_keep, positions=coords_keep, cell=cell_np, pbc=pbc_np)
        write(pruned_path, atoms_pruned, format="cif")
        keep_indices = [str(i) for i, keep in enumerate(keep_mask_full) if keep]
        with open(keep_idx_path, "w", encoding="utf-8") as f:
            f.write(",".join(keep_indices))

        if tis_b is not None:
            try:
                tis_cpu = tis_b.detach().cpu().numpy()
            except Exception:
                tis_cpu = None
            if tis_cpu is not None:
                if tis_cpu.shape[0] == mask_bool.shape[0]:
                    tis_full = tis_cpu[mask_bool]
                else:
                    tis_full = None
                if tis_full is not None and len(tis_full) == len(numbers_full):
                    with open(tis_path, "w", encoding="utf-8") as f:
                        f.write("index,element,x,y,z,tis\n")
                        for idx, (num, coord, tis_val) in enumerate(zip(numbers_full, coords_full, tis_full)):
                            sym = chemical_symbols[int(num)] if int(num) < len(chemical_symbols) else "X"
                            f.write(
                                f"{idx},{sym},{coord[0]:.6f},{coord[1]:.6f},{coord[2]:.6f},{float(tis_val):.6e}\n"
                            )

    def _is_main_process(self):
        return (not self._distributed) or self._global_rank == 0
