# -*- coding: utf-8 -*-


import os
import sys
import subprocess


import random
import time


import argparse


import numpy as np


import torch


import torch.distributed as dist


import yaml


from model import CatGT


from config import Config


from framework import Framework


from transformers import AutoTokenizer


from AdslabData import AdslabDataloader
torch.backends.cuda.matmul.allow_tf32 = False


torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def set_seed(seed: int):


    random.seed(seed)


    np.random.seed(seed)


    torch.manual_seed(seed)


    if torch.cuda.is_available():


        torch.cuda.manual_seed_all(seed)


def _replace_cli_arg(argv, key, value):
    flag = f"--{key}"
    if flag in argv:
        idx = argv.index(flag)
        if idx + 1 < len(argv) and not argv[idx + 1].startswith("--"):
            argv[idx + 1] = str(value)
        else:
            argv.insert(idx + 1, str(value))
    else:
        argv.extend([flag, str(value)])


def init_distributed_mode():


    if not dist.is_available():


        return False, 0, 0, 1


    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:


        return False, 0, 0, 1


    rank = int(os.environ["RANK"])


    world_size = int(os.environ["WORLD_SIZE"])


    local_rank = int(os.environ.get("LOCAL_RANK", 0))


    backend = "nccl" if torch.cuda.is_available() else "gloo"


    if not dist.is_initialized():


        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


    if torch.cuda.is_available():


        torch.cuda.set_device(local_rank)


    return True, rank, local_rank, world_size


def cleanup_distributed_mode():


    if dist.is_available() and dist.is_initialized():


        dist.barrier()


        dist.destroy_process_group()


def _coerce_int(value, field_name):


    if value is None:


        return None


    try:


        return int(value)


    except (TypeError, ValueError):


        try:


            return int(float(value))


        except (TypeError, ValueError) as exc:


            raise ValueError(f"{field_name} expects an integer, got {value}") from exc


def _coerce_float(value, field_name):


    if value is None:


        return None


    try:


        return float(value)


    except (TypeError, ValueError) as exc:


        raise ValueError(f"{field_name} expects a float, got {value}") from exc


def _normalize_plan_entries(entries):


    normalized = []


    for idx, raw_entry in enumerate(entries):


        if not isinstance(raw_entry, dict):


            raise ValueError(f"Plan entry #{idx+1} must be a dict, got {type(raw_entry).__name__}")


        entry = dict(raw_entry)


        epochs = _coerce_int(entry.get("epochs", entry.get("epoch")), f"plan[{idx}].epochs")


        if epochs is None or epochs <= 0:


            continue


        normalized.append({


            "name": entry.get("name") or f"phase_{idx+1}",


            "epochs": epochs,


            "llm_unfreeze_layers": _coerce_int(


                entry.get("llm_unfreeze_layers", entry.get("llm_layers")),


                f"plan[{idx}].llm_unfreeze_layers"


            ),


            "text_unfreeze_layers": _coerce_int(


                entry.get("text_unfreeze_layers", entry.get("text_layers")),


                f"plan[{idx}].text_unfreeze_layers"


            ),


            "learning_rate": _coerce_float(entry.get("learning_rate"), f"plan[{idx}].learning_rate"),


            "lr_scale": _coerce_float(entry.get("lr_scale"), f"plan[{idx}].lr_scale"),


            "weight_decay": _coerce_float(entry.get("weight_decay"), f"plan[{idx}].weight_decay"),


            "text_unfreeze_lr": _coerce_float(


                entry.get("text_unfreeze_lr"),


                f"plan[{idx}].text_unfreeze_lr"


            ),


            "llm_lr_decay": _coerce_float(entry.get("llm_lr_decay"), f"plan[{idx}].llm_lr_decay"),


            "backbone_lr": _coerce_float(


                entry.get("backbone_lr", entry.get("unfrozen_backbone_lr")),


                f"plan[{idx}].backbone_lr"


            ),


            "backbone_lr_scale": _coerce_float(entry.get("backbone_lr_scale"), f"plan[{idx}].backbone_lr_scale"),


            "warmup_ratio": _coerce_float(entry.get("warmup_ratio"), f"plan[{idx}].warmup_ratio"),


            "warmup_steps": _coerce_int(entry.get("warmup_steps"), f"plan[{idx}].warmup_steps"),


        })


    return normalized


def parse_finetune_plan(plan_spec):


    if not plan_spec:


        return []


    spec = plan_spec.strip()


    if not spec:


        return []


    plan_data = None


    candidate_path = os.path.expanduser(spec)


    if os.path.isfile(candidate_path):


        with open(candidate_path, 'r', encoding='utf-8') as f:


            plan_data = yaml.safe_load(f) or []


    else:


        try:


            plan_data = yaml.safe_load(spec)


        except yaml.YAMLError:


            plan_data = None


    if isinstance(plan_data, dict):


        if "plan" in plan_data:


            plan_data = plan_data["plan"]


        elif "phases" in plan_data:


            plan_data = plan_data["phases"]


    if isinstance(plan_data, list):


        return _normalize_plan_entries(plan_data)


    inline_entries = []


    for chunk in spec.split(";"):


        chunk = chunk.strip()


        if not chunk:


            continue


        parts = [p.strip() for p in chunk.split(",") if p.strip()]


        if len(parts) < 4:


            raise ValueError(


                "finetune_plan inline format expects 'epochs,llm_layers,text_layers,lr_scale' entries separated by ';'"


            )


        inline_entries.append({


            "epochs": parts[0],


            "llm_unfreeze_layers": parts[1],


            "text_unfreeze_layers": parts[2],


            "lr_scale": parts[3],


        })


    return _normalize_plan_entries(inline_entries)


def _parse_phase_list_arg(raw_value):


    """Normalize --phase_limit inputs into a clean list or None."""


    if not raw_value:


        return None


    tokens = []


    items = raw_value


    if isinstance(raw_value, str):


        items = [raw_value]


    for entry in items:


        if not entry:


            continue


        parts = [part.strip() for part in entry.split(",")]


        for part in parts:


            if part:


                tokens.append(part)


    return tokens or None


def _parse_phase_value_arg(raw_entries, arg_name):


    """Parse CLI specs like 'phase=1e-3' (can be space/comma separated)."""


    parsed = {}


    if not raw_entries:


        return parsed


    items = raw_entries


    if isinstance(raw_entries, str):


        items = [raw_entries]


    for entry in items:


        if not entry:


            continue


        for chunk in entry.split(","):


            chunk = chunk.strip()


            if not chunk:


                continue


            if "=" not in chunk:


                raise ValueError(f"{arg_name} expects 'phase=value' pairs, got '{chunk}'")


            phase_name, value_str = chunk.split("=", 1)


            phase_name = phase_name.strip()


            value_str = value_str.strip()


            if not phase_name or not value_str:


                raise ValueError(f"{arg_name} expects 'phase=value' pairs, got '{chunk}'")


            parsed[phase_name] = _coerce_float(value_str, f"{arg_name}[{phase_name}]")


    return parsed


def _load_model_state(model, ckpt_path):


    state = torch.load(ckpt_path, map_location="cpu")


    if isinstance(state, dict):


        for key in ("model_state_dict", "state_dict", "model"):


            if key in state:


                state = state[key]


                break


    missing, unexpected = model.load_state_dict(state, strict=False)


    if missing:


        print(f"[Eval] Missing keys: {len(missing)}")


    if unexpected:


        print(f"[Eval] Unexpected keys: {len(unexpected)}")


def run_finetune_plan(


    config,


    framework,


    train_model,


    train_loader,


    val_loader,


    plan_entries,


    resume_state,


    phase_filter=None,


    phase_lr_override=None,


    phase_backbone_override=None,


):


    if not plan_entries:


        return


    base_lr = getattr(config, "base_learning_rate", config.learning_rate)


    base_backbone_lr = getattr(config, "base_unfrozen_backbone_lr", getattr(config, "unfrozen_backbone_lr", None))


    current_state = resume_state


    base_model = train_model.module if hasattr(train_model, "module") else train_model


    total_phases = len(plan_entries)


    allowed_phases = None


    if phase_filter:


        allowed_phases = {name.strip() for name in phase_filter if name.strip()}


    lr_override_map = phase_lr_override or {}


    backbone_override_map = phase_backbone_override or {}


    for idx, phase in enumerate(plan_entries):


        epochs = int(phase.get("epochs", 0))


        if epochs <= 0:


            continue


        phase_name = phase.get("name") or f"phase_{idx+1}"


        if allowed_phases is not None and phase_name not in allowed_phases:


            continue


        config.current_phase = phase_name


        config.resume_mode = "phase"


        config.resume_start_epoch = 0


        config.target_total_epochs = epochs


        config.epochs_to_run = epochs


        config.epoch = epochs


        override_lr = lr_override_map.get(phase_name)


        if override_lr is not None:


            config.learning_rate = float(override_lr)


        elif phase.get("learning_rate") is not None:


            config.learning_rate = float(phase["learning_rate"])


        elif phase.get("lr_scale") is not None:


            config.learning_rate = base_lr * float(phase["lr_scale"])


        else:


            config.learning_rate = base_lr


        override_backbone_lr = backbone_override_map.get(phase_name)


        if override_backbone_lr is not None:


            config.unfrozen_backbone_lr = float(override_backbone_lr)


        elif phase.get("backbone_lr") is not None:


            config.unfrozen_backbone_lr = float(phase["backbone_lr"])


        elif phase.get("backbone_lr_scale") is not None:


            ref_lr = base_backbone_lr if base_backbone_lr is not None else base_lr


            config.unfrozen_backbone_lr = ref_lr * float(phase["backbone_lr_scale"])


        else:


            config.unfrozen_backbone_lr = base_backbone_lr


        if phase.get("weight_decay") is not None:


            config.weight_decay = float(phase["weight_decay"])


        if phase.get("llm_lr_decay") is not None:


            config.llm_lr_decay = float(phase["llm_lr_decay"])


        if phase.get("text_unfreeze_lr") is not None:


            config.text_unfreeze_lr = float(phase["text_unfreeze_lr"])


        phase_warmup_steps = phase.get("warmup_steps")


        phase_warmup_ratio = phase.get("warmup_ratio")


        config.phase_warmup_steps = None


        config.phase_warmup_ratio = None


        if phase_warmup_steps is not None and phase_warmup_steps > 0:


            config.phase_warmup_steps = int(phase_warmup_steps)


        elif phase_warmup_ratio is not None and phase_warmup_ratio > 0:


            config.phase_warmup_ratio = float(phase_warmup_ratio)


        freeze_geom = phase.get("freeze_geom")


        llm_layers = phase.get("llm_unfreeze_layers")


        text_layers = phase.get("text_unfreeze_layers")


        if freeze_geom is not None or llm_layers is not None or text_layers is not None:


            base_model.apply_freeze_settings(


                llm_layers=llm_layers,


                text_layers=text_layers,


                freeze_geom=freeze_geom,


            )


        def _layer_display(value, config_value):


            if value is None or (isinstance(value, (int, float)) and value <= 0):


                return "ALL"


            if value is not None:


                return value


            return config_value


        llm_layers_display = _layer_display(llm_layers, getattr(config, "llm_unfreeze_layers", 0))


        text_layers_display = _layer_display(text_layers, getattr(config, "text_unfreeze_layers", 0))


        geom_freeze_display = freeze_geom if freeze_geom is not None else getattr(config, "freeze_geom", False)


        framework.logging("=" * 60)


        lr_display = f"{config.learning_rate:.2e}"


        if override_lr is not None:


            lr_display += "*"


        backbone_lr_display = (


            config.unfrozen_backbone_lr


            if config.unfrozen_backbone_lr is not None else config.learning_rate


        )


        backbone_display = f"{backbone_lr_display:.2e}"


        if override_backbone_lr is not None:


            backbone_display += "*"


        framework.logging(


            f"Stage-2 Phase {idx+1}/{total_phases} [{phase_name}] "


            f"epochs={epochs}, lr={lr_display}, backbone_lr={backbone_display}, "


            f"llm_layers={llm_layers_display}, text_layers={text_layers_display}, freeze_geom={geom_freeze_display}"


        )


        framework.logging("=" * 60)


        phase_state = None


        if current_state:


            phase_state = dict(current_state)


            phase_state.pop("optimizer_state_dict", None)


        framework.train(train_model, train_loader, val_loader, resume_state=phase_state)


        current_state = None


    config.current_phase = None


    config.phase_warmup_steps = None


    config.phase_warmup_ratio = None


def get_args():


    parser = argparse.ArgumentParser()


    parser.add_argument('--filename', type=str, default="train")


    parser.add_argument('--seed', type=int, default=0)


    parser.add_argument('--num_runs', type=int, default=5,
                        help='Number of random training runs to execute sequentially.')
    parser.add_argument('--device', type=str, default='cuda')


    parser.add_argument('--epoch', type=int, default=50)


    parser.add_argument('--mode', choices=['train', 'val'], default='train',


                        help='Run training or validation only.')
    parser.add_argument('--hidden_nf', type=int, default=128)


    parser.add_argument('--out_node_nf', type=int, default=1)


    parser.add_argument('--in_edge_nf', type=int, default=1)
    parser.add_argument('--bert_name', type=str, default='scibert')


    parser.add_argument('--num_query_token', type=int, default=8)


    parser.add_argument('--cross_attention_freq', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)


    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--dataset_root', type=str,


                        default="./data/oc20data_proc")


    parser.add_argument('--dataset', type=str,  # compat old scripts


                        default="./data/oc20data_proc")


    parser.add_argument('--bert_path', type=str,


                        default="./external/CatBERTa-hf",


                        help='Path to the pretrained text encoder/tokenizer (e.g., SciBERT or CatBERTa).')


    parser.add_argument('--train_file', type=str, default='./data/train.pt')


    parser.add_argument('--val_file', type=str, default='val.pt')


    parser.add_argument('--edge_method', type=str, default='hybrid',


                        choices=['full', 'hybrid'],


                        help='Graph construction method for EGNN (full connectivity or radius+kNN hybrid).')


    parser.add_argument('--edge_cutoff', type=float, default=12.0, help='Cutoff (Angstrom) used when edge_method includes radius-based sparsification.')


    parser.add_argument('--edge_knn', type=int, default=20, help='Number of neighbors per atom when edge_method includes kNN sparsification.')


    parser.add_argument('--ads_cutoff_scale', type=float, default=1.0,


                        help='Multiplier applied to cutoff for adsorbate nodes.')


    parser.add_argument('--ads_knn_scale', type=float, default=1.0,


                        help='Multiplier applied to kNN count for adsorbate nodes.')


    parser.add_argument('--ads_edge_weight', type=float, default=1.0,


                        help='Edge feature multiplier for adsorbate nodes inside EGNN.')


    parser.add_argument('--normalize_labels', dest='normalize_labels', action='store_true',


                        help='Enable target standardization (mean/std) for energy labels.')


    parser.add_argument('--no_normalize_labels', dest='normalize_labels', action='store_false',


                        help='Disable target standardization (v2 default).')


    parser.set_defaults(normalize_labels=False)


    parser.add_argument('--equiformer_root', type=str,


                        default="./external/equiformer_v2",


                        help='Path to equiformer_v2-main/equiformer_v2-main (optional if using repo layout).')


    parser.add_argument('--max_num_elements', type=int, default=90,


                        help='Maximum atomic number for EquiformerV2 embedding table.')
    parser.add_argument('--checkpoint', type=str, default="./checkpoints")


    parser.add_argument('--log_dir', type=str, default="./logs")


    parser.add_argument('--geom_prune_min_tokens', type=int, default=1,


                        help='Minimum number of geometry tokens to keep per sample.')
    parser.add_argument('--geom_prune_min_non_ads', type=int, default=5,


                        help='Minimum number of non-adsorbate tokens to keep per sample.')
    parser.add_argument('--geom_prune_max_non_ads', type=int, default=15,


                        help='Maximum number of non-adsorbate tokens to keep per sample.')

    parser.add_argument('--temperature', type=float, default=1.0,


                        help='Global pruning temperature.')


    parser.add_argument('--geom_distance_reduce', type=str, default='mean',


                        choices=['mean'],


                        help='Distance aggregation from ads atoms (mean only).')


    parser.add_argument('--capture_text_attentions', action='store_true',


                        help='Save text attention maps for debugging.')


    parser.add_argument('--text_attention_dir', type=str, default=None,


                        help='Output directory for text attention dumps.')


    parser.add_argument('--text_attention_interval', type=int, default=0,


                        help='Steps between attention dumps (0 = every step).')
    parser.add_argument('--learning_rate', type=float, default=2e-5)


    parser.add_argument('--amp', dest='amp', action='store_true', help='Enable AMP (mixed precision).')


    parser.add_argument('--no_amp', dest='amp', action='store_false', help='Disable AMP (mixed precision).')


    parser.set_defaults(amp=True)


    parser.add_argument('--early_stop_patience', type=int, default=8)


    parser.add_argument('--early_stop_min_delta', type=float, default=0.0)


    parser.add_argument('--clip_grad_norm', type=float, default=5.0)


    parser.add_argument('--weight_decay', type=float, default=1e-5)


    parser.add_argument('--beta1', type=float, default=0.9)


    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--ema_decay', type=float, default=0.0,
                        help='EMA decay for model weights (0 disables).')


    parser.add_argument('--min_lr', type=float, default=1e-6)


    parser.add_argument('--use_catberta_llrd', dest='use_catberta_llrd', action='store_true',


                        help='?CatBERTa ')


    parser.add_argument('--no_catberta_llrd', dest='use_catberta_llrd', action='store_false')


    parser.set_defaults(use_catberta_llrd=True)


    parser.add_argument('--llm_lr_base', type=float, default=2e-5,


                        help='')


    parser.add_argument('--llm_tier2_scale', type=float, default=1.75,


                        help='CatBERTa  LR ')


    parser.add_argument('--llm_tier3_scale', type=float, default=3.5,


                        help='CatBERTa  LR ')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers.')


    parser.add_argument('--freeze_geom', dest='freeze_geom', action='store_true', help='Freeze geometric encoder parameters.')


    parser.add_argument('--no_freeze_geom', dest='freeze_geom', action='store_false', help='Unfreeze geometric encoder parameters.')


    parser.set_defaults(freeze_geom=True)


    parser.add_argument('--geom_unfreeze_layers', type=int, default=0, help='Number of geometric encoder blocks to unfreeze (from the top).')


    parser.add_argument('--unfrozen_backbone_lr', type=float, default=None, help='Backbone LR when unfrozen.')


    parser.add_argument('--llm_lr_decay', type=float, default=1.0, help='Layerwise LR decay for LLM.')


    parser.add_argument('--llm_unfreeze_layers', type=int, default=999, help='Number of LLM blocks to unfreeze (from the top).')


    parser.add_argument('--text_unfreeze_layers', type=int, default=0, help='Number of text encoder layers to unfreeze (from the top).')


    parser.add_argument('--text_unfreeze_lr', type=float, default=1e-5, help='LR for text encoder layers when unfrozen.')


    parser.add_argument('--temperature_warmup_ratio', type=float, default=0.3, help='Warmup ratio for temperature schedule.')


    parser.add_argument('--run_subdir', type=str, default=None, help='Optional subdir name for checkpoints/logs.')


    parser.add_argument('--capture_geom_importance', action='store_true', help='Dump geometry importance samples per epoch.')
    parser.add_argument('--save_val_preds', action='store_true', help='Save validation predictions/targets for plotting.')
    parser.add_argument('--save_val_preds_path', type=str, default="", help='Optional path for saving val preds (npz).')
    parser.add_argument('--export_pruned_cif_sids', type=str, default='1,2,3,4,5,6',
                        help='Comma-separated sample ids to export pruning CIFs.')
    parser.add_argument('--export_pruned_cif_all', action='store_true',
                        help='Export pruning CIFs for all samples (ignores export_pruned_cif_sids).')
    parser.add_argument('--export_pruned_cif_mode', type=str, default='val',
                        choices=['train', 'val', 'both', 'off'],
                        help='Export pruning CIFs in train/val/both (off disables).')
    parser.add_argument('--export_pruned_cif_dir', type=str, default=None,
                        help='Output directory for pruning CIF exports (default: log_dir/baseline/cif_samples).')
    parser.add_argument('--export_pruned_cif_mask_symbol', type=str, default="",
                        help='If set, write pruned CIF with all atoms but masked ones as this element symbol (e.g., X).')
    parser.add_argument('--capture_gradcam_gnn', action='store_true',
                        help='Enable GradCAM-GNN activation capture.')
    parser.add_argument('--capture_gradcam_llm', action='store_true',
                        help='Enable GradCAM-LLM activation capture.')
    parser.add_argument('--gradcam_min_non_ads', type=int, default=None,
                        help='Minimum non-ads atoms to keep for GradCAM if pruning count is unavailable.')
    parser.add_argument('--gradcam_max_non_ads', type=int, default=None,
                        help='Maximum non-ads atoms to keep for GradCAM if pruning count is unavailable.')
    parser.add_argument('--export_gradcam_dir', type=str, default=None,
                        help='Directory to dump GradCAM keep-index files.')
    parser.add_argument('--export_gradcam_methods', type=str, default='gnn,llm',
                        help='Comma-separated GradCAM methods to export (gnn,llm).')
    parser.add_argument('--export_gradcam_sids', type=str, default='',
                        help='Comma-separated sample indices for GradCAM export.')
    parser.add_argument('--export_gradcam_all', action='store_true',
                        help='Export GradCAM keep-index files for all samples.')
    parser.add_argument('--export_gradcam_first_n', type=int, default=0,
                        help='If >0, export GradCAM outputs for the first N samples only.')
    parser.add_argument('--export_gradcam_cif_keep', action='store_true',
                        help='Export GradCAM keep-only CIFs.')
    parser.add_argument('--export_gradcam_scores', action='store_true',
                        help='Export GradCAM per-atom score files for EP.')
    parser.add_argument('--export_gnnexplainer_dir', type=str, default=None,
                        help='Directory to dump GNNExplainer outputs.')
    parser.add_argument('--export_gnnexplainer_sids', type=str, default='',
                        help='Comma-separated sample indices for GNNExplainer export.')
    parser.add_argument('--export_gnnexplainer_all', action='store_true',
                        help='Export GNNExplainer outputs for all samples.')
    parser.add_argument('--export_gnnexplainer_first_n', type=int, default=0,
                        help='If >0, export GNNExplainer outputs for the first N samples only.')
    parser.add_argument('--gnnexplainer_steps', type=int, default=5,
                        help='Number of optimization steps for GNNExplainer.')
    parser.add_argument('--gnnexplainer_lr', type=float, default=0.1,
                        help='Learning rate for GNNExplainer mask optimization.')
    parser.add_argument('--gnnexplainer_l1', type=float, default=0.05,
                        help='Sparsity weight for GNNExplainer mask optimization.')
    parser.add_argument('--export_gnnexplainer_cif_keep', action='store_true',
                        help='Export GNNExplainer keep-only CIFs.')
    parser.add_argument('--export_gnnexplainer_scores', action='store_true',
                        help='Export GNNExplainer per-atom score files for EP.')
    parser.add_argument('--geom_importance_dump_count', type=int, default=3, help='Samples per epoch for geom importance dumps.')


    parser.add_argument('--probe_steps', type=int, default=0, help='Stop after N steps for probing (0 = full epoch).')


    parser.add_argument('--phase_limit', nargs='*', default=None, help='Limit phases by name.')


    parser.add_argument('--phase_lr_override', nargs='*', default=None, help='Override phase LR: phase=value.')


    parser.add_argument('--phase_backbone_lr_override', nargs='*', default=None, help='Override phase backbone LR: phase=value.')


    parser.add_argument('--auto_two_stage', action='store_true', help='Enable automatic two-stage finetune.')


    parser.add_argument('--two_stage_split', type=int, default=None, help='Epochs for stage1 when auto_two_stage.')


    parser.add_argument('--two_stage_lr_scale', type=float, default=0.5, help='LR scale for stage2 when auto_two_stage.')


    parser.add_argument('--stage1_epochs', type=int, default=None, help='Epochs for stage1_freeze.')


    parser.add_argument('--stage2_epochs', type=int, default=None, help='Epochs for stage2_unfreeze.')


    parser.add_argument('--finetune_plan', type=str, default=None,


                        help='Stage-2 ???? (YAML/JSON ???? "epochs,llm_layers,text_layers,lr_scale;..." ??)')


    parser.add_argument('--resume_config', type=str, default=None, help='Path to an existing config.yaml for resuming')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint (.pth) for resuming')
    parser.add_argument(
        '--resume_mode',
        choices=['auto', 'total', 'additional'],
        default='auto',
        help='How to interpret --epoch when resuming (`total`=target total epochs, `additional`=extra epochs)',
    )

    args = parser.parse_args()
    args._defaults = vars(parser.parse_args([]))
    return args


def main():
    args = get_args()

    num_runs = max(1, int(getattr(args, "num_runs", 1)))
    if args.mode == "train" and num_runs > 1:
        if getattr(args, "resume_checkpoint", None):
            raise ValueError("--num_runs > 1 is only supported for fresh training without --resume_checkpoint")
        if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("--num_runs > 1 is not supported under distributed launch")
        base_run = getattr(args, "run_subdir", None) or getattr(args, "filename", None) or "run"
        base_argv = list(sys.argv[1:])
        for run_idx in range(num_runs):
            run_seed = int(args.seed) + run_idx
            child_argv = list(base_argv)
            _replace_cli_arg(child_argv, "num_runs", 1)
            _replace_cli_arg(child_argv, "seed", run_seed)
            _replace_cli_arg(child_argv, "run_subdir", f"{base_run}_seed_{run_idx + 1:02d}")
            print(f"[MultiRun] Starting run {run_idx + 1}/{num_runs} with seed={run_seed}")
            subprocess.run([sys.executable, os.path.abspath(__file__), *child_argv], check=True)
        return

    if getattr(args, "resume_config", None):
        resume_config_path = os.path.expanduser(args.resume_config)
        if not os.path.isfile(resume_config_path):
            raise FileNotFoundError(f"Resume config not found: {resume_config_path}")
        with open(resume_config_path, 'r', encoding='utf-8') as f:
            loaded_cfg = yaml.safe_load(f) or {}
        defaults = getattr(args, "_defaults", {})
        cli_overrides = set()
        if "--train_file" in sys.argv:
            cli_overrides.add("train_file")
        if "--val_file" in sys.argv:
            cli_overrides.add("val_file")
        for key, value in loaded_cfg.items():
            if key in ('resume_config', 'resume_checkpoint', 'resume_mode'):
                continue
            if key in cli_overrides:
                continue
            if not hasattr(args, key):
                continue
            current_val = getattr(args, key)
            default_val = defaults.get(key)
            if current_val == default_val:
                setattr(args, key, value)
        args.resume_config = resume_config_path


    plan_entries = []


    if getattr(args, "finetune_plan", None):


        try:


            plan_entries = parse_finetune_plan(args.finetune_plan)


        except ValueError as exc:


            raise SystemExit(f"[finetune_plan] {exc}")


    args.finetune_plan_entries = plan_entries


    phase_limit_list = _parse_phase_list_arg(getattr(args, "phase_limit", None))


    args.phase_limit_list = phase_limit_list


    try:


        args.phase_lr_override_map = _parse_phase_value_arg(


            getattr(args, "phase_lr_override", None),


            "phase_lr_override",


        )


        args.phase_backbone_lr_override_map = _parse_phase_value_arg(


            getattr(args, "phase_backbone_lr_override", None),


            "phase_backbone_lr_override",


        )


    except ValueError as exc:


        raise SystemExit(str(exc))
    defaults = getattr(args, "_defaults", {})
    if args.learning_rate == defaults.get("learning_rate"):


        args.learning_rate = args.llm_lr_base
    if args.weight_decay == defaults.get("weight_decay"):


        args.weight_decay = 0.01
    text_default = defaults.get("text_unfreeze_lr", None)


    if text_default is not None and args.text_unfreeze_lr == text_default:


        args.text_unfreeze_lr = max(args.llm_lr_base * 3.6, 1e-6)


    args._base_learning_rate = args.learning_rate


    args._base_unfrozen_backbone_lr = getattr(args, "unfrozen_backbone_lr", None)
    ds_root = args.dataset_root if args.dataset_root else args.dataset


    setattr(args, "dataset_root", ds_root)  # ?Config 


    is_distributed, global_rank, local_rank, world_size = init_distributed_mode()


    if is_distributed:


        if not torch.cuda.is_available():


            raise RuntimeError("?CUDA ")


        args.device = f"cuda:{local_rank}"
    set_seed(args.seed + global_rank)
    config = Config(args)
    config.mode = args.mode


    config.distributed = is_distributed


    config.world_size = world_size


    config.global_rank = global_rank


    config.local_rank = local_rank


    resume_state = None


    resume_checkpoint_path = getattr(args, "resume_checkpoint", None)


    resume_start_epoch = 0


    if resume_checkpoint_path:


        resume_checkpoint_path = os.path.expanduser(resume_checkpoint_path)


        if not os.path.isfile(resume_checkpoint_path):


            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint_path}")


        resume_state = torch.load(resume_checkpoint_path, map_location='cpu')


        resume_start_epoch = int(resume_state.get('epoch', -1)) + 1


        if resume_start_epoch < 0:


            resume_start_epoch = 0


        args.resume_checkpoint = resume_checkpoint_path


    resume_mode = getattr(args, "resume_mode", "auto")


    if args.mode == "val" and not resume_checkpoint_path:


        raise ValueError("--mode val requires --resume_checkpoint")


    if resume_state:


        if resume_mode == "auto":


            resume_mode = "total" if getattr(args, "resume_config", None) else "additional"


        config.resume_mode = resume_mode


        config.resume_start_epoch = resume_start_epoch


        config.resume_checkpoint_path = resume_checkpoint_path


        if resume_mode == "total":


            config.target_total_epochs = config.epoch


            remaining_epochs = max(0, config.target_total_epochs - resume_start_epoch)


            if remaining_epochs <= 0:


                print(f"[Resume] start epoch {resume_start_epoch} already reached target {config.target_total_epochs}.")


                return


            config.epochs_to_run = remaining_epochs


            config.epoch = config.target_total_epochs


        else:


            config.target_total_epochs = resume_start_epoch + config.epoch


            config.epochs_to_run = config.epoch


            config.epoch = config.target_total_epochs


    else:


        config.resume_mode = "none"


        config.resume_start_epoch = 0


        config.resume_checkpoint_path = None


        config.target_total_epochs = config.epoch


        config.epochs_to_run = config.epoch
    config.learning_rate = args.learning_rate


    config.weight_decay = args.weight_decay


    config.beta1 = args.beta1


    config.beta2 = args.beta2

    mode_dir = (

        getattr(args, "run_subdir", None)

        or getattr(args, "filename", None)

        or "run"

    )

    checkpoint_mode_dir = os.path.join(config.checkpoint, mode_dir)

    # Print training config

    # Print training config

    if (not is_distributed) or global_rank == 0:

        print("\n" + "=" * 60)

        phase_label = "Validation" if args.mode == "val" else "Training"
        print(f"[Start {phase_label}]")
        print("=" * 60)
        if args.mode == "val":
            print("Mode: Validation")
        else:
            print("Mode: Baseline (no pruning)")
        if resume_state:
            print(f"[Resume] epoch {resume_start_epoch} ({resume_mode})")
        print("=" * 60 + "\n")

    log_mode_dir = os.path.join(config.log_dir, mode_dir)

    config.save_subdir = mode_dir

    config.log_subdir = mode_dir

    config.checkpoint_active_dir = checkpoint_mode_dir

    config.log_active_dir = log_mode_dir


    os.makedirs(config.checkpoint, exist_ok=True)

    os.makedirs(config.log_dir, exist_ok=True)

    os.makedirs(checkpoint_mode_dir, exist_ok=True)

    os.makedirs(log_mode_dir, exist_ok=True)


    model = CatGT(config).to(config.device)

    if ((not is_distributed) or global_rank == 0) and args.mode != 'val':

        model.print_model_params()


    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer._num_workers = config.num_workers

    if hasattr(model, "register_text_tokenizer"):

        model.register_text_tokenizer(tokenizer)


    train_model = model

    if is_distributed:

        device_ids = [local_rank] if torch.cuda.is_available() else None

        train_model = torch.nn.parallel.DistributedDataParallel(

            model,

            device_ids=device_ids,

            output_device=local_rank if torch.cuda.is_available() else None,

            find_unused_parameters=True,

        )


    if (not is_distributed) or global_rank == 0:

        config.save_config(checkpoint_mode_dir)


    dm = AdslabDataloader(


        batch_size=config.batch_size,


        root=config.dataset_root,


        text_max_len=config.max_len,


        device=config.device,


        tokenizer=tokenizer,


        train_file=config.train_file,


        val_file=config.val_file,


        distributed=is_distributed,


        world_size=world_size,


        rank=global_rank,


        edge_method=config.edge_method,


        edge_cutoff=config.edge_cutoff,


        edge_knn=config.edge_knn,


        ads_cutoff_scale=config.ads_cutoff_scale,


        ads_knn_scale=config.ads_knn_scale,


        normalize_labels=config.normalize_labels,


    )
    # Print training config


    if (not is_distributed) or global_rank == 0:


        print("\n" + "=" * 60)


        phase_label = "Validation" if args.mode == "val" else "Training"
        print(f"[Start {phase_label}]")
        print("=" * 60)
        if args.mode == "val":
            print("Mode: Validation")
        else:
            print("Mode: Baseline (no pruning)")
        if resume_state:
            print(f"[Resume] epoch {resume_start_epoch} ({resume_mode})")
        print("=" * 60 + "\n")
    framework = Framework(config)


    train_loader = dm.train_dataloader()


    val_loader = dm.val_dataloader()


    if args.mode == "val":

        _load_model_state(model, resume_checkpoint_path)

        model.eval()

        if framework._is_main_process():

            base_save_path = getattr(config, "save_val_preds_path", "") or ""
            if base_save_path:
                root, ext = os.path.splitext(base_save_path)
                if not ext:
                    ext = ".npz"
                no_prune_path = f"{root}_noprun{ext}"
                prune_path = f"{root}_prune{ext}"
            else:
                no_prune_path = ""
                prune_path = ""

            start_time = time.perf_counter()
            if no_prune_path:
                config.save_val_preds_path = no_prune_path
            else:
                config.save_val_preds_path = ""
            avg_val, metrics = framework._run_validation(
                model,
                val_loader,
                config.temperature,
                config.val_subset_ratio,
                epoch_idx=0,
                desc="Val",
                pruning_mode="disable",
            )
            elapsed = time.perf_counter() - start_time
            print(
                f"[Val] MAE: {avg_val:.6f} | RMSE: {metrics['rmse']:.6f} | R2: {metrics['r2']:.6f} | Time: {elapsed:.2f}s"
            )

        if is_distributed:

            cleanup_distributed_mode()

        return


    if config.finetune_plan:


        run_finetune_plan(


            config,


            framework,


            train_model,


            train_loader,


            val_loader,


            config.finetune_plan,


            resume_state,


            phase_filter=getattr(config, "phase_limit", None),


            phase_lr_override=getattr(config, "phase_lr_overrides", None),


            phase_backbone_override=getattr(config, "phase_backbone_lr_overrides", None),


        )


    else:


        framework.train(train_model, train_loader, val_loader, resume_state=resume_state)


    if is_distributed:


        cleanup_distributed_mode()


if __name__ == "__main__":


    main()


