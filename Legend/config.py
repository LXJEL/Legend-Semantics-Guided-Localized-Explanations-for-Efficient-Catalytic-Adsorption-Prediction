import os
import yaml

class Config(object):
    def __init__(self, args):
        # ===== Training hyperparameters =====
        self.max_len = args.max_len
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.device = args.device
        self.bert_path = getattr(args, 'bert_path', "./external/CatBERTa-hf")

        self.iter_num = 1000
        self.block_size = 2048
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.dropout = 0.2
        self.bias = False

        # Optimizer & scheduler
        self.learning_rate = getattr(args, "learning_rate", 1e-5)
        self.amp = bool(getattr(args, "amp", False))
        self.early_stop_patience = getattr(args, "early_stop_patience", 8)
        self.early_stop_min_delta = getattr(args, "early_stop_min_delta", 0.0)
        self.clip_grad_norm = getattr(args, "clip_grad_norm", 5.0)
        self.max_iters = 600000
        self.weight_decay = 1e-4
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.min_lr = 1e-7
        self.unfrozen_backbone_lr = getattr(args, "unfrozen_backbone_lr", None)
        self.llm_lr_decay = getattr(args, "llm_lr_decay", 1.0)
        self.use_catberta_llrd = getattr(args, "use_catberta_llrd", False)
        self.llm_lr_base = getattr(args, "llm_lr_base", None)
        self.llm_tier2_scale = getattr(args, "llm_tier2_scale", 1.75)
        self.llm_tier3_scale = getattr(args, "llm_tier3_scale", 3.5)
        # Head/attention LR scale; use 3.6 for CatBERTa style, otherwise keep legacy defaults
        if getattr(args, "use_catberta_lr_all", False):
            self.attention_lr_scale = 3.6
            self.new_layer_lr_scale = 3.6
        else:
            self.attention_lr_scale = getattr(args, "attention_lr_scale", 5.0)
            self.new_layer_lr_scale = getattr(args, "new_layer_lr_scale", 3.0)
        self.period = 100
        self.temperature = getattr(args, "temperature", 1.0)
        self.base_learning_rate = getattr(args, "_base_learning_rate", self.learning_rate)
        self.base_unfrozen_backbone_lr = getattr(
            args, "_base_unfrozen_backbone_lr", self.unfrozen_backbone_lr
        )
        self.phase_warmup_ratio = None
        self.phase_warmup_steps = None
        self.ema_decay = float(getattr(args, "ema_decay", 0.0))

        # ===== Pruning =====
        self.enable_pruning = True
        self.geom_prune_min_tokens = getattr(args, "geom_prune_min_tokens", 1)
        # Sharper distance bias; lower default sigma
        self.geom_distance_sigma = 3.0
        self.geom_distance_flat = 5.0
        # Use mean distance aggregation by default
        self.geom_distance_reduce = getattr(args, "geom_distance_reduce", "mean")
        self.cross_attn_repeats = getattr(args, 'cross_attn_repeats', 3)
        self.freeze_geom = getattr(args, "freeze_geom", False)
        self.geom_unfreeze_layers = getattr(args, "geom_unfreeze_layers", 2)
        self.text_unfreeze_layers = getattr(args, "text_unfreeze_layers", 2)
        self.text_unfreeze_lr = getattr(args, "text_unfreeze_lr", 3e-5)
        self.llm_unfreeze_layers = getattr(args, 'llm_unfreeze_layers', 2)
        self.capture_text_attentions = getattr(args, 'capture_text_attentions', False)
        self.text_attention_dir = getattr(args, 'text_attention_dir', None)
        self.text_attention_interval = getattr(args, 'text_attention_interval', 0)
        self.val_subset_ratio = getattr(args, 'val_subset_ratio', 1.0)
        self.temperature_warmup_ratio = getattr(args, 'temperature_warmup_ratio', 0.3)
        self.run_subdir = getattr(args, 'run_subdir', None)
        self.finetune_plan = getattr(args, "finetune_plan_entries", [])
        self.phase_limit = getattr(args, "phase_limit_list", None)
        self.phase_lr_overrides = getattr(args, "phase_lr_override_map", {})
        self.phase_backbone_lr_overrides = getattr(args, "phase_backbone_lr_override_map", {})
        self.current_phase = None
        self.probe_steps = max(0, int(getattr(args, "probe_steps", 0)))

        # ===== EGNN (keep your defaults) =====
        self.gtm = True
        self.out_node_nf = args.out_node_nf
        self.in_edge_nf = args.in_edge_nf
        self.hidden_nf = args.hidden_nf

        # ===== Projector / text encoder =====
        self.projection_dim = 256
        self.bert_name = args.bert_name
        self.num_query_token = args.num_query_token
        self.cross_attention_freq = args.cross_attention_freq

        # ===== Data paths (default to oc20data_proc) =====
        # Root dir: can be overridden by --dataset_root; otherwise use the absolute path
        self.dataset_root = getattr(
            args, "dataset_root",
            "./external/oc20data_proc"
        )
        # Split file names (can be overridden by CLI)
        self.train_file = getattr(args, "train_file", "./data/train.pt")
        self.val_file   = getattr(args, "val_file",   "val.pt")
        self.edge_method = getattr(args, "edge_method", "hybrid")
        self.edge_cutoff = getattr(args, "edge_cutoff", 12.0)
        self.edge_knn = getattr(args, "edge_knn", 20)
        self.ads_cutoff_scale = getattr(args, "ads_cutoff_scale", 1.0)
        self.ads_knn_scale = getattr(args, "ads_knn_scale", 1.0)
        self.ads_edge_weight = getattr(args, "ads_edge_weight", 1.0)
        self.normalize_labels = getattr(args, "normalize_labels", False)
        self.max_num_elements = getattr(args, "max_num_elements", 90)
        self.equiformer_root = getattr(
            args,
            "equiformer_root",
            "./external/equiformer_v2-main/equiformer_v2-main",
        )
        # EquiformerV2 (77M) defaults
        self.equiformer_num_layers = getattr(args, "equiformer_num_layers", 12)
        self.equiformer_sphere_channels = getattr(args, "equiformer_sphere_channels", 128)
        self.equiformer_attn_hidden_channels = getattr(args, "equiformer_attn_hidden_channels", 64)
        self.equiformer_num_heads = getattr(args, "equiformer_num_heads", 8)
        self.equiformer_attn_alpha_channels = getattr(args, "equiformer_attn_alpha_channels", 64)
        self.equiformer_attn_value_channels = getattr(args, "equiformer_attn_value_channels", 16)
        self.equiformer_ffn_hidden_channels = getattr(args, "equiformer_ffn_hidden_channels", 128)
        self.equiformer_norm_type = getattr(args, "equiformer_norm_type", "layer_norm_sh")
        self.equiformer_lmax_list = getattr(args, "equiformer_lmax_list", [6])
        self.equiformer_mmax_list = getattr(args, "equiformer_mmax_list", [2])
        self.equiformer_grid_resolution = getattr(args, "equiformer_grid_resolution", 18)
        self.equiformer_num_sphere_samples = getattr(args, "equiformer_num_sphere_samples", 128)
        self.equiformer_edge_channels = getattr(args, "equiformer_edge_channels", 128)
        self.equiformer_use_atom_edge_embedding = getattr(args, "equiformer_use_atom_edge_embedding", True)
        self.equiformer_share_atom_edge_embedding = getattr(args, "equiformer_share_atom_edge_embedding", False)
        self.equiformer_distance_function = getattr(args, "equiformer_distance_function", "gaussian")
        self.equiformer_num_distance_basis = getattr(args, "equiformer_num_distance_basis", 512)
        self.equiformer_attn_activation = getattr(args, "equiformer_attn_activation", "silu")
        self.equiformer_use_s2_act_attn = getattr(args, "equiformer_use_s2_act_attn", False)
        self.equiformer_use_attn_renorm = getattr(args, "equiformer_use_attn_renorm", True)
        self.equiformer_ffn_activation = getattr(args, "equiformer_ffn_activation", "silu")
        self.equiformer_use_gate_act = getattr(args, "equiformer_use_gate_act", False)
        self.equiformer_use_grid_mlp = getattr(args, "equiformer_use_grid_mlp", True)
        self.equiformer_use_sep_s2_act = getattr(args, "equiformer_use_sep_s2_act", True)
        self.equiformer_alpha_drop = getattr(args, "equiformer_alpha_drop", 0.1)
        self.equiformer_drop_path_rate = getattr(args, "equiformer_drop_path_rate", 0.05)
        self.equiformer_proj_drop = getattr(args, "equiformer_proj_drop", 0.0)
        self.equiformer_weight_init = getattr(args, "equiformer_weight_init", "uniform")
        self.equiformer_ckpt_path = getattr(
            args,
            "equiformer_ckpt_path",
            "./external/equiformer_v2-main/equiformer_v2-main/checkpoints/100/checkpoints/2026-01-17-23-57-52/best_checkpoint.pt",
        )

        # Backward compatibility: if external code uses config.dataset, treat it as root
        self.dataset = self.dataset_root
        # Logging & checkpoints
        self.checkpoint = args.checkpoint
        self.log_dir = args.log_dir
        # Use _pa suffix for per-atom metrics
        self.log_save_name = getattr(args, "log_save_name", "oc20_catgt.json")
        # Whether to log geometry-importance stats
        self.capture_geom_importance = getattr(args, "capture_geom_importance", False)
        self.geom_importance_dump_count = getattr(args, "geom_importance_dump_count", 3)
        self.save_val_preds = bool(getattr(args, "save_val_preds", False))
        self.save_val_preds_path = str(getattr(args, "save_val_preds_path", "") or "").strip()
        self.export_pruned_cif_mask_symbol = str(
            getattr(args, "export_pruned_cif_mask_symbol", "") or ""
        ).strip()
        self.geom_prune_min_non_ads = getattr(args, "geom_prune_min_non_ads", None)
        self.geom_prune_max_non_ads = getattr(args, "geom_prune_max_non_ads", None)
        self.capture_gradcam_gnn = bool(getattr(args, "capture_gradcam_gnn", False))
        self.capture_gradcam_llm = bool(getattr(args, "capture_gradcam_llm", False))
        self.gradcam_min_non_ads = getattr(args, "gradcam_min_non_ads", None)
        self.gradcam_max_non_ads = getattr(args, "gradcam_max_non_ads", None)
        raw_gradcam_sids = getattr(args, "export_gradcam_sids", "")
        self.export_gradcam_all = bool(getattr(args, "export_gradcam_all", False))
        gradcam_sids = []
        if raw_gradcam_sids:
            if isinstance(raw_gradcam_sids, (list, tuple)):
                tokens = raw_gradcam_sids
            else:
                tokens = str(raw_gradcam_sids).replace(";", ",").split(",")
            for token in tokens:
                token = str(token).strip()
                if not token:
                    continue
                try:
                    gradcam_sids.append(int(float(token)))
                except (TypeError, ValueError):
                    continue
        self.export_gradcam_sids = gradcam_sids
        self.export_gradcam_dir = getattr(args, "export_gradcam_dir", None)
        self.export_gradcam_methods = str(getattr(args, "export_gradcam_methods", "gnn,llm") or "")
        self.export_gradcam_first_n = int(getattr(args, "export_gradcam_first_n", 0) or 0)
        self.export_gradcam_cif_keep = bool(getattr(args, "export_gradcam_cif_keep", False))
        self.export_gradcam_scores = bool(getattr(args, "export_gradcam_scores", False))
        self.export_gnnexplainer_dir = getattr(args, "export_gnnexplainer_dir", None)
        self.export_gnnexplainer_all = bool(getattr(args, "export_gnnexplainer_all", False))
        self.export_gnnexplainer_first_n = int(getattr(args, "export_gnnexplainer_first_n", 0) or 0)
        self.gnnexplainer_steps = int(getattr(args, "gnnexplainer_steps", 5))
        self.gnnexplainer_lr = float(getattr(args, "gnnexplainer_lr", 0.1))
        self.gnnexplainer_l1 = float(getattr(args, "gnnexplainer_l1", 0.05))
        self.export_gnnexplainer_cif_keep = bool(getattr(args, "export_gnnexplainer_cif_keep", False))
        self.export_gnnexplainer_scores = bool(getattr(args, "export_gnnexplainer_scores", False))
        raw_gnnexplainer_sids = getattr(args, "export_gnnexplainer_sids", "")
        gnnexplainer_sids = []
        if raw_gnnexplainer_sids:
            if isinstance(raw_gnnexplainer_sids, (list, tuple)):
                tokens = raw_gnnexplainer_sids
            else:
                tokens = str(raw_gnnexplainer_sids).replace(";", ",").split(",")
            for token in tokens:
                token = str(token).strip()
                if not token:
                    continue
                try:
                    gnnexplainer_sids.append(int(float(token)))
                except (TypeError, ValueError):
                    continue
        self.export_gnnexplainer_sids = gnnexplainer_sids
        raw_cif_sids = getattr(args, "export_pruned_cif_sids", "")
        self.export_pruned_cif_all = bool(getattr(args, "export_pruned_cif_all", False))
        cif_sids = []
        if raw_cif_sids:
            if isinstance(raw_cif_sids, (list, tuple)):
                tokens = raw_cif_sids
            else:
                tokens = str(raw_cif_sids).replace(";", ",").split(",")
            for token in tokens:
                token = str(token).strip()
                if not token:
                    continue
                try:
                    cif_sids.append(int(float(token)))
                except (TypeError, ValueError):
                    continue
        self.export_pruned_cif_sids = cif_sids
        self.export_pruned_cif_mode = getattr(args, "export_pruned_cif_mode", "off")
        self.export_pruned_cif_dir = getattr(args, "export_pruned_cif_dir", None)

        # DataLoader settings
        self.num_workers = getattr(args, "num_workers", 4)

    # Convenience helpers for full paths
    def train_path(self):
        return os.path.join(self.dataset_root, self.train_file)

    def val_path(self):
        return os.path.join(self.dataset_root, self.val_file)

    def save_config(self, save_dir, filename="config.yaml"):
        """config.yaml"""
        os.makedirs(save_dir, exist_ok=True)
        config_dict = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith('_')
            and k
            not in [
                'ctx',
                'ptdtype',
            ]
        }
        with open(os.path.join(save_dir, filename), 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True)
