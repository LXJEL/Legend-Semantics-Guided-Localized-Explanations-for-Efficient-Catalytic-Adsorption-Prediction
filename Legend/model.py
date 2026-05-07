# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from equiformer.equiformer_adapter import EquiformerV2Adapter
from torch.nn import functional as F
from transformers import RobertaConfig, RobertaModel
import numpy as np
import math
import einops


# ==================== MADTP ====================
def vector_gather(vectors, indices):
    """
    MADTP?
    """
    N, L, D = vectors.shape
    squeeze = False
    if indices.ndim == 1:
        squeeze = True
        indices = indices.unsqueeze(-1)
    N2, K = indices.shape
    assert N == N2
    indices = einops.repeat(indices, "N K -> N K D", D=D)
    out = torch.gather(vectors, dim=1, index=indices)
    if squeeze:
        out = out.squeeze(1)
    return out


class Sparsemax(nn.Module):
    """MADTP sparsemax activation."""

    def __init__(self, dim=None):
        super(Sparsemax, self).__init__()
        self.dim = -1 if dim is None else dim

    def forward(self, input):
        device = input.device
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=device, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        zs_sparse = is_gt * zs
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        output = torch.max(torch.zeros_like(input), input - taus)
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output


class Query_model(nn.Module):
    """MADTP query model for token-space dictionary interactions."""

    def __init__(self, ft_dim, sd_dim, temperature=1, att_func_type='softmax', pool_type='sum', map_func=True):
        super().__init__()

        assert att_func_type in ['softmax', 'sigmoid', 'sparsemax']
        self.att_func_type = att_func_type

        assert pool_type in ['mean', 'max', 'sum']
        self.pool_type = pool_type

        if self.att_func_type == 'softmax':
            self.att_activation = nn.Softmax(dim=-1)
        elif self.att_func_type == 'sparsemax':
            self.att_activation = Sparsemax(dim=-1)
        else:
            self.att_activation = nn.Sigmoid()

        self.att_dim = sd_dim
        self.temperature = temperature
        self.map_func = map_func

        # Tokenpace dictionary?
        if self.map_func:
            self.q_map = nn.Sequential(
                nn.Linear(ft_dim, sd_dim),
            )

    def forward(self, ft, sd, mask=None, return_token_att=False, temperature=1):
        """
        Args:
            ft: [batch, token_num, ft_dim] - 
            sd: [FDT_num, sd_dim] - space dictionary
            mask: [batch, token_num] - padding mask
            return_token_att: token?
            temperature: 
        """
        # space_dict?
        sd = sd.to(ft.device)

        # oken?
        if self.map_func:
            q = self.q_map(ft)  # [batch, token_num, sd_dim]
        else:
            q = ft

        k = sd.unsqueeze(0)  # [1, sd_num, sd_dim]
        k = k.transpose(2, 1)  # [1, sd_dim, sd_num]
        inner_dot = torch.matmul(q, k)  # [batch, token_num, sd_num]

        if return_token_att:
            token_att = inner_dot

        inner_dot = inner_dot / math.sqrt(self.att_dim)  # 
        att_weight = torch.softmax(inner_dot.permute(0, 2, 1) / temperature, dim=-1)  # [batch, sd_num, token_num]
        att_ft = torch.bmm(att_weight, q)  # [batch, sd_num, sd_dim]

        if return_token_att:
            return token_att, att_ft, sd

        return att_weight, att_ft, sd


# ---------------------- 1. ?(? ----------------------
class Pruner(nn.Module):
    """Reusable token pruner that can operate on arbitrary feature dimensions."""

    def __init__(self, token_dim, sd_dim):
        super().__init__()
        self.query_model = Query_model(
            ft_dim=token_dim,
            sd_dim=sd_dim,
            att_func_type='sparsemax',
            pool_type='sum',
            map_func=True,
        )
        self.min_keep_tokens = 1
        # Optional constraints for non-mandatory (e.g., non-adsorbate) tokens.
        self.min_keep_non_ads = None
        self.max_keep_non_ads = None
        self.sparsemax = Sparsemax(dim=-1)
        self._threshold_printed = False
        self._temp_warn_printed = False

    def _normalize_temperature(self, temperature):
        if temperature is None:
            return 0.0
        temp = float(temperature)
        if temp < 0:
            temp = 0.0
        if temp > 1 and not self._temp_warn_printed:
            print(f"[Pruner] temperature {temp:.3f} > 1.0, clamping to 1.0 to match MADTP.")
            self._temp_warn_printed = True
        return max(0.0, min(temp, 1.0))
        self.reset_cache()

    def reset_cache(self):
        self.last_importance = None
        self.last_raw_importance = None
        self.last_importance_bias = None
        self.last_keep_indices = None
        self.last_keep_mask = None
        self.last_keep_lengths = None
        self.last_kept_tokens = None
        self.last_original_length = None
        self.last_padded_length = None
        self.last_merge_sources = None
        self.last_mandatory_mask = None

    def forward(self, x, space_dict=None, temperature=0.0, importance_bias=None,
                mandatory_keep_mask=None, attention_scores=None,
                tis_scores=None, query_mask=None):
        return self.prune(
            x,
            space_dict,
            temperature,
            importance_bias=importance_bias,
            mandatory_keep_mask=mandatory_keep_mask,
            attention_scores=attention_scores,
            tis_scores=tis_scores,
            query_mask=query_mask,
        )

    def prune(self, x, space_dict=None, temperature=0.0, importance_bias=None,
              mandatory_keep_mask=None, valid_mask=None,
              attention_scores=None, tis_scores=None, query_mask=None):
        batch_size, seq_len, hidden_dim = x.shape
        self.last_original_length = seq_len

        norm_temp = self._normalize_temperature(temperature)
        can_use_query = (space_dict is not None) and (norm_temp > 0)
        if (not can_use_query) and importance_bias is None:
            keep_indices = [torch.arange(seq_len, device=x.device) for _ in range(batch_size)]
            keep_masks = torch.ones(batch_size, seq_len, device=x.device, dtype=torch.bool)
            keep_lengths = torch.full((batch_size,), seq_len, device=x.device, dtype=torch.long)

            self.last_keep_indices = [idx.detach().cpu().tolist() for idx in keep_indices]
            self.last_keep_mask = keep_masks.detach().cpu()
            self.last_keep_lengths = keep_lengths.detach().cpu()
            self.last_merge_sources = [{} for _ in range(batch_size)]
            self.last_kept_tokens = [x[b].detach().cpu() for b in range(batch_size)]
            self.last_importance = None
            self.last_raw_importance = None
            self.last_importance_bias = None
            self.last_mandatory_mask = None
            self.last_padded_length = seq_len
            return x, keep_indices, keep_lengths

        use_threshold = (attention_scores is not None) and (tis_scores is not None)
        if not use_threshold:
            raise ValueError("Pruner now requires attention_scores and tis_scores for MADTP-style pruning.")
        threshold_keep_mask = None

        if tis_scores is not None:
            tis_scores = tis_scores.to(x.device, dtype=x.dtype)
            if tis_scores.shape[0] != batch_size:
                raise ValueError("tis_scores batch size mismatch with tokens")
            if tis_scores.shape[1] != seq_len:
                if tis_scores.shape[1] < seq_len:
                    pad_len = seq_len - tis_scores.shape[1]
                    pad = torch.zeros(batch_size, pad_len, device=tis_scores.device, dtype=tis_scores.dtype)
                    tis_scores = torch.cat([tis_scores, pad], dim=1)
                else:
                    tis_scores = tis_scores[:, :seq_len]
            combined_importance = tis_scores
        else:
            combined_importance = None

        if can_use_query and not use_threshold:
            token_att, _, _ = self.query_model(
                ft=x,
                sd=space_dict,
                return_token_att=True,
                temperature=norm_temp
            )

            global_importance = token_att.max(dim=-1)[0]
            token_norms = torch.norm(x, dim=-1)
            local_importance = token_norms / (token_norms.max(dim=-1, keepdim=True)[0] + 1e-8)

            spatial_importance = torch.var(token_att, dim=-1)
            spatial_importance = spatial_importance / (spatial_importance.max(dim=-1, keepdim=True)[0] + 1e-8)

            combined_importance = (
                0.3 * global_importance +
                0.4 * local_importance +
                0.3 * spatial_importance
            )
        if use_threshold:
            attention_scores = attention_scores.to(x.device, dtype=x.dtype)
            if attention_scores.dim() != 3:
                attention_scores = attention_scores.view(batch_size, -1, seq_len)
            if attention_scores.shape[2] != seq_len:
                if attention_scores.shape[2] < seq_len:
                    pad_len = seq_len - attention_scores.shape[2]
                    pad = torch.zeros(
                        batch_size,
                        attention_scores.shape[1],
                        pad_len,
                        device=attention_scores.device,
                        dtype=attention_scores.dtype,
                    )
                    attention_scores = torch.cat([attention_scores, pad], dim=2)
                else:
                    attention_scores = attention_scores[:, :, :seq_len]
            if query_mask is None:
                query_mask = torch.ones(
                    batch_size,
                    attention_scores.shape[1],
                    device=x.device,
                    dtype=torch.bool,
                )
            else:
                query_mask = query_mask.to(x.device, dtype=torch.bool)
                if query_mask.shape[1] != attention_scores.shape[1]:
                    if query_mask.shape[1] < attention_scores.shape[1]:
                        pad_len = attention_scores.shape[1] - query_mask.shape[1]
                        pad = torch.zeros(batch_size, pad_len, device=query_mask.device, dtype=torch.bool)
                        query_mask = torch.cat([query_mask, pad], dim=1)
                    else:
                        query_mask = query_mask[:, :attention_scores.shape[1]]
            attn_for_threshold = attention_scores
            query_mask_for_threshold = query_mask
        else:
            attn_for_threshold = None
            query_mask_for_threshold = None

        if valid_mask is not None:
            valid_mask = valid_mask.to(x.device, dtype=torch.bool)
            if valid_mask.shape != (batch_size, seq_len):
                valid_mask = torch.nn.functional.pad(
                    valid_mask, (0, max(0, seq_len - valid_mask.shape[1])), value=False
                )
                valid_mask = valid_mask[:, :seq_len]
        else:
            valid_mask = torch.ones(batch_size, seq_len, device=x.device, dtype=torch.bool)

        if use_threshold:
            threshold_keep_mask = self._compute_threshold_mask(
                attn_for_threshold, combined_importance, query_mask_for_threshold, valid_mask, norm_temp
            )

        if importance_bias is not None:
            device_ref = x.device if combined_importance is None else combined_importance.device
            dtype_ref = x.dtype if combined_importance is None else combined_importance.dtype
            importance_bias = importance_bias.to(device_ref, dtype=dtype_ref)
            ref_len = combined_importance.shape[1] if combined_importance is not None else seq_len
            if importance_bias.dim() == 2 and importance_bias.shape[1] != ref_len:
                need_len = ref_len
                cur_len = importance_bias.shape[1]
                if cur_len < need_len:
                    pad = torch.zeros(
                        importance_bias.shape[0],
                        need_len - cur_len,
                        device=importance_bias.device,
                        dtype=importance_bias.dtype,
                    )
                    importance_bias = torch.cat([importance_bias, pad], dim=1)
                else:
                    importance_bias = importance_bias[:, :need_len]
            elif importance_bias.dim() != 2 and combined_importance is not None:
                importance_bias = importance_bias.view_as(combined_importance)
            importance_bias = importance_bias.clamp(min=0.0)
            if combined_importance is None:
                combined_importance = importance_bias + 1e-8
            else:
                combined_importance = combined_importance * (1.0 + importance_bias)

        if combined_importance is None:
            combined_importance = torch.ones(batch_size, seq_len, device=x.device, dtype=x.dtype)
        combined_importance = combined_importance * valid_mask.float()
        importance_sums = combined_importance.sum(dim=-1, keepdim=True)
        zero_rows = importance_sums <= 1e-8
        if zero_rows.any():
            combined_importance = combined_importance + zero_rows.float() * 1e-8
            importance_sums = combined_importance.sum(dim=-1, keepdim=True)

        temp_val = norm_temp
        sharp_factor = max(1.0, min(max(temp_val, 1e-6) / 3.0, 6.0))
        combined_importance = combined_importance.clamp(min=1e-8).pow(sharp_factor)
        combined_importance = combined_importance / (combined_importance.sum(dim=-1, keepdim=True) + 1e-8)
        raw_importance = combined_importance.clone()

        selected_tokens = []
        keep_indices_list = []
        keep_masks = []
        keep_lengths = []
        merge_mappings = []

        for b in range(batch_size):
            valid_mask_b = valid_mask[b]
            valid_idx = torch.nonzero(valid_mask_b, as_tuple=True)[0]
            if valid_idx.numel() == 0:
                keep_indices_list.append(torch.tensor([], device=x.device, dtype=torch.long))
                keep_masks.append(torch.zeros(seq_len, dtype=torch.bool, device=x.device))
                keep_lengths.append(0)
                merge_mappings.append({})
                continue
            importance_full = combined_importance[b]
            importance_vals = importance_full[valid_idx]
            keep_mask_b = threshold_keep_mask[b]
            keep_mask_b = keep_mask_b & valid_mask_b
            mandatory_mask_b = None
            mandatory_indices = None
            if mandatory_keep_mask is not None:
                mandatory_mask_b = mandatory_keep_mask[b]
                if mandatory_mask_b.shape[0] != seq_len:
                    if mandatory_mask_b.shape[0] < seq_len:
                        pad_len = seq_len - mandatory_mask_b.shape[0]
                        pad = torch.zeros(pad_len, device=mandatory_mask_b.device, dtype=mandatory_mask_b.dtype)
                        mandatory_mask_b = torch.cat([mandatory_mask_b, pad], dim=0)
                    else:
                        mandatory_mask_b = mandatory_mask_b[:seq_len]
                mandatory_mask_b = mandatory_mask_b.bool()
                mandatory_indices = torch.nonzero(mandatory_mask_b, as_tuple=True)[0]
                keep_mask_b = keep_mask_b | mandatory_mask_b
            else:
                mandatory_mask_b = torch.zeros(seq_len, dtype=torch.bool, device=x.device)
            min_keep = max(1, int(getattr(self, "min_keep_tokens", 1)))
            if min_keep > 1:
                cur_keep = int(keep_mask_b.sum().item())
                if cur_keep < min_keep:
                    scores = importance_full.clone()
                    scores = scores.masked_fill(~valid_mask_b, float("-inf"))
                    valid_count = int(torch.isfinite(scores).sum().item())
                    topk = min(min_keep, max(1, valid_count))
                    top_idx = torch.topk(scores, k=topk).indices
                    keep_mask_b = keep_mask_b.clone()
                    keep_mask_b[top_idx] = True
            non_ads_mask_b = valid_mask_b & ~mandatory_mask_b
            min_non_ads = getattr(self, "min_keep_non_ads", None)
            max_non_ads = getattr(self, "max_keep_non_ads", None)
            if min_non_ads is not None:
                try:
                    min_non_ads = int(min_non_ads)
                except (TypeError, ValueError):
                    min_non_ads = None
            if max_non_ads is not None:
                try:
                    max_non_ads = int(max_non_ads)
                except (TypeError, ValueError):
                    max_non_ads = None
            if min_non_ads is not None and max_non_ads is not None and max_non_ads < min_non_ads:
                max_non_ads = min_non_ads
            if min_non_ads is not None and min_non_ads > 0:
                valid_non_ads = int(non_ads_mask_b.sum().item())
                target = min(min_non_ads, valid_non_ads)
                cur_non_ads = int((keep_mask_b & non_ads_mask_b).sum().item())
                if target > 0 and cur_non_ads < target:
                    scores = importance_full.clone()
                    scores = scores.masked_fill(~non_ads_mask_b, float("-inf"))
                    top_idx = torch.topk(scores, k=target).indices
                    keep_mask_b = keep_mask_b.clone()
                    keep_mask_b[top_idx] = True
            if max_non_ads is not None and max_non_ads >= 0:
                valid_non_ads = int(non_ads_mask_b.sum().item())
                target = min(max_non_ads, valid_non_ads)
                cur_non_ads = int((keep_mask_b & non_ads_mask_b).sum().item())
                if cur_non_ads > target:
                    scores = importance_full.clone()
                    scores = scores.masked_fill(~non_ads_mask_b, float("-inf"))
                    if target > 0:
                        top_idx = torch.topk(scores, k=target).indices
                        keep_mask_b = keep_mask_b & ~non_ads_mask_b
                        keep_mask_b[top_idx] = True
                    else:
                        keep_mask_b = keep_mask_b & ~non_ads_mask_b
            keep_indices = torch.nonzero(keep_mask_b, as_tuple=True)[0]
            if keep_indices.numel() == 0:
                best_local = torch.argmax(importance_vals)
                keep_indices = valid_idx[best_local : best_local + 1]
            mandatory_cpu = mandatory_indices.detach().cpu().tolist() if mandatory_indices is not None else []

            keep_indices = keep_indices.sort()[0]

            if mandatory_cpu:
                keep_set = set(keep_indices.detach().cpu().tolist())
                keep_set.update(mandatory_cpu)
                keep_indices = torch.tensor(sorted(keep_set), device=x.device, dtype=torch.long)

            merge_map = {int(idx.item()): [] for idx in keep_indices}

            keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=x.device)
            keep_mask[keep_indices] = True
            pruned_indices = torch.nonzero(~keep_mask, as_tuple=True)[0]

            x_b = x[b]
            kept_tokens = x_b[keep_indices]
            importance_kept = importance_full[keep_indices]

            if pruned_indices.numel() > 0:
                pruned_tokens = x_b[pruned_indices]
                pruned_weights = importance_full[pruned_indices]

                similarities = torch.matmul(pruned_tokens, kept_tokens.T)
                closest_indices = torch.argmax(similarities, dim=1)

                weighted_pruned_tokens = pruned_tokens * pruned_weights.unsqueeze(-1)
                aggregated_vectors = torch.zeros_like(kept_tokens)
                aggregated_weights = torch.zeros_like(importance_kept)

                index_expanded = closest_indices.unsqueeze(-1).expand(-1, hidden_dim)
                aggregated_vectors.scatter_add_(0, index_expanded, weighted_pruned_tokens)
                aggregated_weights.scatter_add_(0, closest_indices, pruned_weights)

                total_weights = importance_kept + aggregated_weights + 1e-8
                kept_tokens = (importance_kept.unsqueeze(-1) * kept_tokens + aggregated_vectors) / total_weights.unsqueeze(-1)

                closest_list = closest_indices.tolist()
                pruned_list = pruned_indices.tolist()
                for src_idx, dest_rel in zip(pruned_list, closest_list):
                    dest_idx = int(keep_indices[dest_rel].item())
                    merge_map.setdefault(dest_idx, []).append(int(src_idx))

            selected_tokens.append(kept_tokens)
            keep_indices_list.append(keep_indices)
            keep_masks.append(keep_mask)
            keep_lengths.append(keep_indices.numel())
            merge_mappings.append({k: v for k, v in merge_map.items() if v})

        max_keep_len = max(keep_lengths) if keep_lengths else seq_len
        pruned_x = torch.zeros(batch_size, max_keep_len, hidden_dim, device=x.device, dtype=x.dtype)

        for b, tokens in enumerate(selected_tokens):
            keep_len = tokens.shape[0]
            if keep_len == 0:
                continue
            pruned_x[b, :keep_len] = tokens

        self.last_keep_indices = [idx.detach().cpu().tolist() for idx in keep_indices_list]
        self.last_keep_mask = torch.stack(keep_masks).detach().cpu()
        self.last_keep_lengths = torch.tensor(keep_lengths, dtype=torch.long)
        self.last_merge_sources = merge_mappings
        self.last_kept_tokens = [tokens.detach().cpu() for tokens in selected_tokens]
        self.last_importance = combined_importance.detach().cpu() if combined_importance is not None else None
        self.last_raw_importance = raw_importance.detach().cpu() if raw_importance is not None else None
        self.last_importance_bias = importance_bias.detach().cpu() if importance_bias is not None else None
        self.last_mandatory_mask = mandatory_keep_mask.detach().cpu().clone() if mandatory_keep_mask is not None else None
        self.last_padded_length = max_keep_len

        return pruned_x, keep_indices_list, keep_lengths

    def _compute_threshold_mask(self, attention_scores, tis_scores, query_mask, valid_mask, temperature):
        if attention_scores is None or tis_scores is None:
            return None
        scale = max(self._normalize_temperature(temperature), 1e-6)
        scaled = attention_scores * scale
        if query_mask is not None:
            scaled = scaled.masked_fill(~query_mask.unsqueeze(-1), float("-inf"))
        sparse_attn = self.sparsemax(scaled)
        if query_mask is not None:
            sparse_attn = sparse_attn * query_mask.unsqueeze(-1).to(dtype=sparse_attn.dtype)
        if sparse_attn.dim() == 3:
            if query_mask is not None:
                query_count = query_mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=sparse_attn.dtype)
                text_mean = sparse_attn.sum(dim=1) / query_count
            else:
                text_mean = sparse_attn.mean(dim=1)
        else:
            text_mean = sparse_attn
        weighted = text_mean * tis_scores
        theta = weighted.min(dim=-1).values
        finite_mask = torch.isfinite(theta)
        fallback = tis_scores.min(dim=-1).values
        theta = torch.where(finite_mask, theta, fallback - 1e-6)
        keep_mask = tis_scores > theta.unsqueeze(-1)
        if valid_mask is not None:
            keep_mask = keep_mask & valid_mask
        row_counts = keep_mask.sum(dim=-1)
        zero_rows = row_counts == 0
        if zero_rows.any():
            masked_tis = tis_scores
            if valid_mask is not None:
                masked_tis = masked_tis.masked_fill(~valid_mask, float("-inf"))
            top_idx = masked_tis.argmax(dim=-1)
            batch_idx = torch.arange(tis_scores.size(0), device=tis_scores.device)
            keep_mask[batch_idx[zero_rows], top_idx[zero_rows]] = True
        if not self._threshold_printed:
            keep_ratio = keep_mask.float().sum(dim=-1) / keep_mask.size(-1)
            theta_cpu = theta.detach().cpu()
            print(
                f"[PruneThreshold] temp_scale={scale:.3f} "
                f"theta_mean={theta_cpu.mean():.4f} theta_min={theta_cpu.min():.4f} "
                f"keep_mean={keep_ratio.mean().item():.3f} keep_min={keep_ratio.min().item():.3f} "
                f"keep_max={keep_ratio.max().item():.3f}"
            )
            self._threshold_printed = True
        return keep_mask


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim=371, num_heads=7, sd_dim=768):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.sd_dim = sd_dim
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
        )

        self.query = nn.Parameter(torch.randn(1, hidden_dim) * 0.01)
        self.proj_q = nn.Linear(hidden_dim, hidden_dim)
        self.proj_k = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.token_pruner = None
        self.prune_proj = nn.Linear(hidden_dim, sd_dim)
        self.criterion = nn.CosineEmbeddingLoss()

        self._attn_printed = False
        self.attention_map = None
        self.cls_attention = None


    def forward(self, x, space_dict=None, temperature=0):
        batch_size, seq_len, hidden_dim = x.shape

        q = self.proj_q(self.query.expand(batch_size, -1))
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)   # [B,H,1,hd]
        k = self.proj_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,N,hd]

        attn_score = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B,H,1,N]
        attn_weight = F.softmax(attn_score, dim=-1)
        self.attention_map = attn_weight  # [B,H,1,N]
        self.cls_attention = attn_weight.mean(dim=1).squeeze(1)  # [B,N]

        v = x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,N,hd]
        attn_out = torch.matmul(attn_weight, v)  # [B,H,1,hd]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, 1, hidden_dim)  # [B,1,C]

        attn_out = self.out_proj(attn_out)
        attn_out = self.layer_norm(attn_out)
        cls_feature = attn_out.squeeze(1)  # [B,C]

        attn_weight_flat = attn_weight.mean(dim=1).squeeze(1)  # [B,N]
        pruned_feature = None
        if temperature > 0 and space_dict is not None:
            pruned_feature = self.prune_proj(cls_feature)  # [B, sd_dim]

        return cls_feature, attn_weight_flat, pruned_feature

# ---------------------- 2. ?----------------------
class Projector(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=1024, output_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.proj(x)


class Mlpadin(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc21 = nn.Linear(hidden_features, hidden_features)
        self.fc22 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    @staticmethod
    def calc_mean_std(f, eps=1e-5):
        b, n, c = f.shape
        f_var = f.var(dim=1) + eps
        f_std = f_var.sqrt().view(b, 1, c)
        f_mean = f.mean(dim=1).view(b, 1, c)
        return ((f - f_mean.expand(f.size())) / f_std.expand(f.size())).transpose(1, 2)

    def forward(self, text_ir, text):
        xc = self.fc1(text_ir)
        xc1 = self.act(xc)
        xc = self.drop(xc1)
        b = self.fc21(xc)
        g = self.fc22(xc)
        ans = self.calc_mean_std(text) * (1 + g) + b
        return ans


# ---------------------- 3.  ----------------------
class CatGT(nn.Module):
    def __init__(self, config):
        super(CatGT, self).__init__()
        self.config = config
        self.device = config.device
        self.sd_num = getattr(config, 'sd_num', 100)
        self.sd_dim = getattr(config, 'sd_dim', 768)
        self.hidden_nf = getattr(config, 'hidden_nf', 128)
        self.space_dict = nn.Parameter(torch.randn(self.sd_num, self.sd_dim))
        self.criterion = nn.CosineEmbeddingLoss()

        self.text_encoder, text_hidden = self.init_text_encoder(
            getattr(config, "bert_path", None), self.device
        )
        self.qformer_model = self.text_encoder

        self.geometric_encoder = self.init_geometric_encoder(config, self.device)
        self.hidden_nf = getattr(self.geometric_encoder, "hidden_nf", self.hidden_nf)
        setattr(self.config, "hidden_nf", self.hidden_nf)
        self.ln_geometric = nn.LayerNorm(self.hidden_nf).to(self.device)

        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden, 256 * 4),
            nn.GELU(),
            nn.Linear(256 * 4, 256),
        ).to(self.device)

        self.qformer_trainable_layer_ids = []

        hidden_dim_llm = getattr(config, "n_embd", 768)
        self.new_wte = nn.Sequential(
            nn.Linear(256, hidden_dim_llm),
            nn.LayerNorm(hidden_dim_llm)
        ).to(self.device)

        self.geom_proj_to_hidden = nn.Sequential(
            nn.Linear(self.hidden_nf, hidden_dim_llm),
            nn.LayerNorm(hidden_dim_llm)
        ).to(self.device)
        self.geom_adapter = nn.Sequential(
            nn.Linear(hidden_dim_llm, hidden_dim_llm),
            nn.GELU(),
            nn.Linear(hidden_dim_llm, hidden_dim_llm),
        ).to(self.device)
        attn_pool_heads = getattr(config, "attn_pool_heads", 8)
        if hidden_dim_llm % attn_pool_heads != 0:
            attn_pool_heads = 8
        self.text2geom_attn = nn.MultiheadAttention(embed_dim=hidden_dim_llm, num_heads=attn_pool_heads, batch_first=True).to(self.device)
        self.geom2text_attn = nn.MultiheadAttention(embed_dim=hidden_dim_llm, num_heads=attn_pool_heads, batch_first=True).to(self.device)
        self.geom_self_attn = nn.MultiheadAttention(embed_dim=hidden_dim_llm, num_heads=attn_pool_heads, batch_first=True).to(self.device)
        self.type_embedding = nn.Embedding(2, hidden_dim_llm).to(self.device)

        self.attention_pool = AttentionPooling(hidden_dim=hidden_dim_llm, num_heads=attn_pool_heads, sd_dim=self.sd_num).to(self.device)
        self.geom_token_pruner = Pruner(token_dim=hidden_dim_llm, sd_dim=self.sd_num).to(self.device)
        self.enable_pruning = True
        self.sequence_pruning_enabled = False
        self.pre_token_pruner = None
        self._capture_text_attentions = bool(getattr(config, "capture_text_attentions", False))
        self.last_text_attentions = None
        self._text_attention_interval = 0
        self._text_attention_dir = None
        if self._capture_text_attentions:
            self._text_attention_interval = getattr(config, "text_attention_interval", 0)
            if self._text_attention_interval <= 0:
                self._text_attention_interval = 1
            attn_dir = getattr(config, "text_attention_dir", None)
            if not attn_dir:
                log_dir = getattr(config, "log_active_dir", getattr(config, "log_dir", "."))
                attn_dir = os.path.join(log_dir, "qformer_attn")
            os.makedirs(attn_dir, exist_ok=True)
            self._text_attention_dir = attn_dir

        #  flush
        # Paper energy head: scalar node energy prediction over the pruned motif.
        self.geom_energy_head = nn.Sequential(
            nn.LayerNorm(hidden_dim_llm),
            nn.Linear(hidden_dim_llm, hidden_dim_llm),
            nn.GELU(),
            nn.Linear(hidden_dim_llm, 1),
        ).to(self.device)
        self._pre_prune_printed = False
        self.last_pre_pruning_info = None

        self.target_mean = None
        self.target_std = None
        self.last_s_token_pre = None
        self.last_token_dist = None
        # default trainable layer ids
        self.llm_trainable_layer_ids = []
        self.qformer_trainable_layer_ids = []
        self._geom_importance_prints = 0
        self.target_stats_loaded = False
        self._pre_keep_ratio = 1.0
        self.geom_distance_sigma = max(float(getattr(config, "geom_distance_sigma", 3.0)), 1e-4)
        self.cross_attn_repeats = max(1, int(getattr(config, "cross_attn_repeats", 3)))
        self.geom_prune_min_tokens = max(1, int(getattr(config, "geom_prune_min_tokens", 1)))
        self.geom_prune_min_non_ads = getattr(config, "geom_prune_min_non_ads", None)
        self.geom_prune_max_non_ads = getattr(config, "geom_prune_max_non_ads", None)
        self._capture_gradcam_gnn = bool(getattr(config, "capture_gradcam_gnn", False))
        self._capture_gradcam_llm = bool(getattr(config, "capture_gradcam_llm", False))
        self.last_gradcam_text_act = None
        self.last_text2geom_attn = None
        self.last_text_mask = None
        self.explainer_mask = None
        self.enable_explainer_mask = False
        # ?epoch ?        self._geom_importance_logged_epoch = None
        self._current_epoch = None
        #  mean
        self.config.geom_distance_reduce = "mean"
        self.geom_token_pruner.min_keep_tokens = self.geom_prune_min_tokens
        if self.geom_prune_min_non_ads is not None and int(self.geom_prune_min_non_ads) > 0:
            self.geom_token_pruner.min_keep_non_ads = int(self.geom_prune_min_non_ads)
        if self.geom_prune_max_non_ads is not None and int(self.geom_prune_max_non_ads) >= 0:
            self.geom_token_pruner.max_keep_non_ads = int(self.geom_prune_max_non_ads)
        self.last_s_cls = None
        self.last_s_self = None
        self.last_s_token = None
        self.last_tis_pre = None
        self.last_tis = None

        # Apply module-specific freezing (geometry/text).
        self.apply_module_freeze()

    @classmethod
    def init_text_encoder(cls, bert_path, device):
        if not bert_path:
            raise ValueError("bert_path is required to initialize the text encoder.")
        # CatBERTa is Roberta-based; load explicitly to avoid auto-model fallback.
        config = RobertaConfig.from_pretrained(bert_path)
        model = RobertaModel.from_pretrained(bert_path, config=config)
        model.to(device)
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(model.config, "hidden_size")
        return model, int(hidden_size)

    @classmethod
    def init_geometric_encoder(cls, config, device):
        eq_root = getattr(config, "equiformer_root", None)
        max_len = getattr(config, "max_len", 128)
        max_num_elements = int(getattr(config, "max_num_elements", 100))
        hidden_nf = int(getattr(config, "equiformer_sphere_channels", getattr(config, "hidden_nf", 128)))
        max_radius = float(getattr(config, "edge_cutoff", 4.5))
        equiformer_kwargs = dict(
            num_layers=int(getattr(config, "equiformer_num_layers", 12)),
            attn_hidden_channels=int(getattr(config, "equiformer_attn_hidden_channels", 64)),
            num_heads=int(getattr(config, "equiformer_num_heads", 8)),
            attn_alpha_channels=int(getattr(config, "equiformer_attn_alpha_channels", 64)),
            attn_value_channels=int(getattr(config, "equiformer_attn_value_channels", 16)),
            ffn_hidden_channels=int(getattr(config, "equiformer_ffn_hidden_channels", 128)),
            norm_type=getattr(config, "equiformer_norm_type", "layer_norm_sh"),
            lmax_list=getattr(config, "equiformer_lmax_list", [6]),
            mmax_list=getattr(config, "equiformer_mmax_list", [2]),
            grid_resolution=getattr(config, "equiformer_grid_resolution", 18),
            num_sphere_samples=int(getattr(config, "equiformer_num_sphere_samples", 128)),
            edge_channels=int(getattr(config, "equiformer_edge_channels", 128)),
            use_atom_edge_embedding=bool(getattr(config, "equiformer_use_atom_edge_embedding", True)),
            share_atom_edge_embedding=bool(getattr(config, "equiformer_share_atom_edge_embedding", False)),
            distance_function=getattr(config, "equiformer_distance_function", "gaussian"),
            num_distance_basis=int(getattr(config, "equiformer_num_distance_basis", 512)),
            attn_activation=getattr(config, "equiformer_attn_activation", "silu"),
            use_s2_act_attn=bool(getattr(config, "equiformer_use_s2_act_attn", False)),
            use_attn_renorm=bool(getattr(config, "equiformer_use_attn_renorm", True)),
            ffn_activation=getattr(config, "equiformer_ffn_activation", "silu"),
            use_gate_act=bool(getattr(config, "equiformer_use_gate_act", False)),
            use_grid_mlp=bool(getattr(config, "equiformer_use_grid_mlp", True)),
            use_sep_s2_act=bool(getattr(config, "equiformer_use_sep_s2_act", True)),
            alpha_drop=float(getattr(config, "equiformer_alpha_drop", 0.1)),
            drop_path_rate=float(getattr(config, "equiformer_drop_path_rate", 0.05)),
            proj_drop=float(getattr(config, "equiformer_proj_drop", 0.0)),
            weight_init=getattr(config, "equiformer_weight_init", "uniform"),
            ckpt_path=getattr(config, "equiformer_ckpt_path", None),
        )
        encoder = EquiformerV2Adapter(
            equiformer_root=eq_root,
            hidden_nf=hidden_nf,
            max_len=max_len,
            max_num_elements=max_num_elements,
            max_radius=max_radius,
            **equiformer_kwargs,
        ).to(device)
        return encoder

    # ----------  ----------
    def freeze_components(self):
        self.apply_module_freeze()

    def apply_module_freeze(self):
        # Start from all trainable, then apply per-module freezes.
        for _, param in self.named_parameters():
            param.requires_grad = True

        if getattr(self.config, "freeze_geom", False) and self.geometric_encoder is not None:
            for p in self.geometric_encoder.parameters():
                p.requires_grad = False
            geom_layers = getattr(self.config, "geom_unfreeze_layers", 0)
            try:
                geom_layers = int(geom_layers)
            except (TypeError, ValueError):
                geom_layers = 0
            if geom_layers > 0:
                encoder = getattr(self.geometric_encoder, "encoder", None)
                model = getattr(encoder, "model", None) if encoder is not None else None
                blocks = getattr(model, "blocks", None) if model is not None else None
                if blocks is not None:
                    total = len(blocks)
                    start = max(0, total - geom_layers)
                    for idx in range(start, total):
                        for p in blocks[idx].parameters():
                            p.requires_grad = True

        text_layers = getattr(self.config, "text_unfreeze_layers", 0)
        try:
            text_layers = int(text_layers)
        except (TypeError, ValueError):
            text_layers = 0
        if self.text_encoder is not None:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            if text_layers > 0 and hasattr(self.text_encoder, "encoder"):
                encoder_layers = getattr(self.text_encoder, "encoder", None)
                if encoder_layers and hasattr(encoder_layers, "layer"):
                    total = len(encoder_layers.layer)
                    start = max(0, total - text_layers)
                    for idx, layer in enumerate(encoder_layers.layer):
                        if idx >= start:
                            for p in layer.parameters():
                                p.requires_grad = True

    def _freeze_all_but_heads(self):
        keep_prefixes = (
            "attention_pool",
            "new_wte",
            "geom_proj_to_hidden",
            "geom_adapter",
            "text_proj",
            "text2geom_attn",
            "geom2text_attn",
            "type_embedding",
        )
        for name, param in self.named_parameters():
            keep = (
                any(name.startswith(prefix) for prefix in keep_prefixes)
                or self._is_trainable_qformer_param(name)
            )
            param.requires_grad = keep
        if hasattr(self, "space_dict"):
            self.space_dict.requires_grad = False

    def _get_qformer_layers(self):
        if self.text_encoder is None:
            return None
        encoder = getattr(self.text_encoder, "encoder", None)
        if encoder is None:
            return None
        return getattr(encoder, "layer", None)

    def _unfreeze_qformer_layers(self, num_layers):
        layers = self._get_qformer_layers()
        if layers is None or num_layers <= 0:
            return []
        total = len(layers)
        num_layers = min(num_layers, total)
        trainable_ids = list(range(total - num_layers, total))
        for idx in trainable_ids:
            for p in layers[idx].parameters():
                p.requires_grad = True
        return trainable_ids

    def apply_freeze_settings(self, llm_layers=None, text_layers=None, freeze_geom=None):
        updated = False
        if freeze_geom is not None:
            freeze_geom_flag = bool(freeze_geom)
            if freeze_geom_flag != bool(getattr(self.config, "freeze_geom", False)):
                self.config.freeze_geom = freeze_geom_flag
                updated = True
        if text_layers is not None:
            text_layers = max(0, int(text_layers))
            current_text_layers = int(
                getattr(self.config, "text_unfreeze_layers", 0)
            )
            if text_layers != current_text_layers:
                self.config.text_unfreeze_layers = text_layers
                updated = True

        if not updated:
            return False

        self.apply_module_freeze()
        return True

    def _is_trainable_qformer_param(self, name):
        # Default: unfreeze last N layers (config.qformer_trainable_layer_ids overrides).
        if getattr(self, "qformer_trainable_layer_ids", None):
            layer_ids = self.qformer_trainable_layer_ids
        else:
            layers = self._get_qformer_layers()
            total = len(layers) if layers is not None else 0
            keep = max(
                0,
                int(
                    getattr(self.config, "text_unfreeze_layers", 8)
                ),
            )
            keep = min(keep, total)
            start = max(0, total - keep)
            layer_ids = list(range(start, total))
        for idx in layer_ids:
            if f"text_encoder.encoder.layer.{idx}." in name:
                return True
        return False

    def dump_text_attentions(self, batch_sids, phase, step):
        if not self._capture_text_attentions:
            return
        if self.last_text_attentions is None or not self.last_text_attentions:
            return
        if self._text_attention_dir is None:
            return
        interval = max(1, getattr(self, "_text_attention_interval", 1))
        if step % interval != 0:
            return
        try:
            payload = {
                "phase": phase,
                "step": int(step),
                "sids": batch_sids,
                "attentions": [
                    att.detach().to(dtype=torch.float16, device="cpu")
                    for att in self.last_text_attentions
                ],
            }
            filename = f"{phase}_step{int(step):06d}.pt"
            torch.save(payload, os.path.join(self._text_attention_dir, filename))
        except Exception as exc:
            print(f"[AttentionDump] : {exc}")
        finally:
            self.last_text_attentions = None
    def print_model_params(self):
        total = 0
        trainable = 0
        non_trainable = 0

        print("\n[Model Parameters]")
        for name, param in self.named_parameters():
            cnt = param.numel()
            total += cnt
            if param.requires_grad:
                trainable += cnt
                print(f"[trainable] {name:45s} -> {cnt/1e3:6.1f}K")
            else:
                non_trainable += cnt
                print(f"[frozen]    {name:45s} -> {cnt/1e3:6.1f}K")

        print("\n[Parameter Summary]")
        print(f"Total params:     {total/1e6:6.2f}M")
        print(f"Trainable params: {trainable/1e6:6.2f}M")
        print(f"Frozen params:    {non_trainable/1e6:6.2f}M\n")

    def register_text_tokenizer(self, tokenizer):
        """
        Legacy hook kept for compatibility. No-op since text token priorities were removed.
        """
        return


    # ---------- ?----------
    def configure_optimizers(self, weight_decay, learning_rate, betas):
        decay = []
        no_decay = []
        qformer_decay = []
        qformer_no_decay = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue

            name_l = name.lower()
            is_no_decay = (
                name.endswith("bias")
                or "layernorm" in name_l or ".ln" in name_l or "ln." in name_l or "norm" in name_l
                or "embedding" in name_l or "embeddings" in name_l
            )

            is_qformer = self._is_trainable_qformer_param(name)

            if is_qformer:
                if is_no_decay:
                    qformer_no_decay.append(p)
                else:
                    qformer_decay.append(p)
            elif is_no_decay:
                no_decay.append(p)
            else:
                decay.append(p)

        optim_groups = []
        if decay:
            optim_groups.append({"params": decay, "lr": learning_rate, "weight_decay": weight_decay})
        if no_decay:
            optim_groups.append({"params": no_decay, "lr": learning_rate, "weight_decay": 0.0})
        qformer_lr = getattr(self.config, "text_unfreeze_lr", min(learning_rate, 1e-5))
        if qformer_decay:
            optim_groups.append({
                "params": qformer_decay,
                "lr": qformer_lr,
                "weight_decay": weight_decay,
            })
        if qformer_no_decay:
            optim_groups.append({
                "params": qformer_no_decay,
                "lr": qformer_lr,
                "weight_decay": 0.0,
            })
        return torch.optim.AdamW(optim_groups, betas=betas)

    # ----------  ----------
    def forward(self, h, x, edges, strings, mask, target, pos_mask=None, temperature=0, return_feature=False, train=True):
        # ? BatchEncoding / tuple
        if isinstance(strings, tuple) and len(strings) == 2:
            strings = {"input_ids": strings[0], "attention_mask": strings[1]}

        text_input_ids = None
        text_attention_mask = None
        if isinstance(strings, dict):
            text_input_ids = strings.get("input_ids", None)
            text_attention_mask = strings.get("attention_mask", None)
        else:
            text_input_ids = getattr(strings, "input_ids", None)
            text_attention_mask = getattr(strings, "attention_mask", None)
        self.last_pre_pruning_info = None
        self._sequence_reduction_stats = 0.0
        if not self.target_stats_loaded:
            if not getattr(self.config, "normalize_labels", False):
                self.target_mean = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                self.target_std = torch.tensor(1.0, device=self.device, dtype=torch.float32)
                self.target_stats_loaded = True
            else:
                stats_root = getattr(self.config, "dataset_root", None) or getattr(self.config, "dataset", None)
                assert stats_root is not None, "Config dataset_root dataset"
                stats_path = os.path.join(stats_root, "target_stats.pt")
                assert os.path.exists(stats_path), f"{stats_path}"

                stats = torch.load(stats_path, map_location=self.device)
                self.target_mean = torch.tensor(stats["mean"], device=self.device, dtype=torch.float32)
                self.target_std  = torch.tensor(stats["std"],  device=self.device, dtype=torch.float32)
                self.target_stats_loaded = True
                print(f"[] : {self.target_mean.item():.4f}, : {self.target_std.item():.4f}")

        batch_size = h.shape[0]
        if h.dim() == 3 and h.size(-1) == 1:
            h = h.squeeze(-1)

        #  ( FP32  AMP )
        amp_enabled = torch.is_autocast_enabled()
        h_geo = h
        x_geo = x
        if amp_enabled:
            x_geo = x_geo.to(dtype=torch.float32)
        with torch.cuda.amp.autocast(enabled=False):
            batch_node = self.geometric_encoder(h_geo, x_geo, edges, mask, batch_size, pos_mask=pos_mask)
            batch_node = self.ln_geometric(batch_node)
        if amp_enabled:
            batch_node = batch_node.to(dtype=h.dtype)

        graph_seq_len = batch_node.size(1)
        if mask.size(1) != graph_seq_len:
            if mask.size(1) > graph_seq_len:
                mask = mask[:, :graph_seq_len]
            else:
                pad = torch.zeros(
                    mask.size(0),
                    graph_seq_len - mask.size(1),
                    device=mask.device,
                    dtype=mask.dtype,
                )
                mask = torch.cat([mask, pad], dim=1)
        mask_bool = mask.to(dtype=torch.bool)
        if pos_mask is not None:
            if pos_mask.dim() == 1:
                pos_mask = pos_mask.unsqueeze(0)
            if pos_mask.size(1) != graph_seq_len:
                if pos_mask.size(1) > graph_seq_len:
                    pos_mask = pos_mask[:, :graph_seq_len]
                else:
                    pad = torch.zeros(
                        pos_mask.size(0),
                        graph_seq_len - pos_mask.size(1),
                        device=pos_mask.device,
                        dtype=pos_mask.dtype,
                    )
                    pos_mask = torch.cat([pos_mask, pad], dim=1)

        self.last_gradcam_gnn_act = None
        if self._capture_gradcam_gnn and torch.is_grad_enabled():
            batch_node = batch_node.detach().requires_grad_(True)
            self.last_gradcam_gnn_act = batch_node

        batch_node = batch_node * mask.unsqueeze(-1).float()
        if self.enable_explainer_mask and self.explainer_mask is not None:
            exp_mask = self.explainer_mask
            if exp_mask.dim() == 1:
                exp_mask = exp_mask.unsqueeze(0)
            if exp_mask.size(1) != batch_node.size(1):
                if exp_mask.size(1) > batch_node.size(1):
                    exp_mask = exp_mask[:, :batch_node.size(1)]
                else:
                    pad = torch.zeros(
                        exp_mask.size(0),
                        batch_node.size(1) - exp_mask.size(1),
                        device=exp_mask.device,
                        dtype=exp_mask.dtype,
                    )
                    exp_mask = torch.cat([exp_mask, pad], dim=1)
            batch_node = batch_node * exp_mask.unsqueeze(-1).to(dtype=batch_node.dtype)

        # Qformer 
        # Q-Former query aggregation removed; keep per-atom tokens.

        # ?query_token ?
        capture_text_attn = getattr(self, "_capture_text_attentions", False)
        if isinstance(strings, dict):
            token_emb = self.text_encoder(
                input_ids=strings["input_ids"],
                attention_mask=strings["attention_mask"],
                return_dict=True,
                output_attentions=capture_text_attn,
            )
            text_mask = strings["attention_mask"].to(dtype=torch.bool)
        else:
            token_emb = self.text_encoder(
                input_ids=strings.input_ids,
                attention_mask=strings.attention_mask,
                return_dict=True,
                output_attentions=capture_text_attn,
            )
            text_mask = strings.attention_mask.to(dtype=torch.bool)
        if self._capture_gradcam_llm:
            self.last_text_mask = text_mask.detach()
        else:
            self.last_text_mask = None
        text_rep = F.normalize(self.text_proj(token_emb.last_hidden_state), dim=-1)
        if self._capture_gradcam_llm and torch.is_grad_enabled():
            text_rep = text_rep.detach().requires_grad_(True)
            self.last_gradcam_text_act = text_rep
        else:
            self.last_gradcam_text_act = None
        if capture_text_attn and hasattr(token_emb, "attentions"):
            self.last_text_attentions = tuple(att.detach() for att in token_emb.attentions)
        else:
            self.last_text_attentions = None

        geom_hidden = self.geom_proj_to_hidden(batch_node)  # [B, G, 1024]
        geom_hidden = geom_hidden + self.geom_adapter(geom_hidden)
        text_hidden = self.new_wte(text_rep)               # [B, T, 1024]
        geom_context = geom_hidden
        text_context = text_hidden
        geom_importance = None
        text2geom_attn_map = None
        s_self = None
        s_token = None

        #  ->  cross-attention?        # MultiheadAttention  key_padding_mask ?True=
        geom_pad_mask = ~mask_bool
        text_mask_bool = text_mask.bool()
        text_pad_mask = ~text_mask_bool
        repeat_times = max(1, int(getattr(self, "cross_attn_repeats", 1)))

        #  S_self
        self_attn_out, self_attn_weights = self.geom_self_attn(
            geom_context,
            geom_context,
            geom_context,
            key_padding_mask=geom_pad_mask,
            need_weights=True,
        )
        geom_context = geom_context + self_attn_out
        if self_attn_weights is not None:
            if self_attn_weights.dim() == 4:
                self_attn_mean = self_attn_weights.mean(dim=1)
            else:
                self_attn_mean = self_attn_weights
            if self_attn_mean.dim() == 4:
                self_attn_mean = self_attn_mean.mean(dim=1)
            s_self_vals = self_attn_mean.max(dim=-1).values
            s_self = s_self_vals / (s_self_vals.sum(dim=1, keepdim=True) + 1e-8)
        else:
            s_self = None

        importance_bias = None
        # Distance bias disabled (uniform attention weights).
        mandatory_mask = None
        ads_mask = None
        token_distance_cache = [None] * batch_size
        if pos_mask is not None:
            if pos_mask.dim() == 1:
                pos_mask = pos_mask.unsqueeze(0)
            pos_mask = pos_mask.to(mask.device, dtype=torch.bool)
            ads_mask = pos_mask & mask_bool
            if ads_mask.any():
                mandatory_mask = ads_mask
                for b in range(batch_size):
                    valid_idx = torch.nonzero(mask_bool[b], as_tuple=True)[0]
                    if valid_idx.numel() == 0:
                        continue
                    ads_idx = torch.nonzero(ads_mask[b], as_tuple=True)[0]
                    if ads_idx.numel() == 0:
                        continue
                    token_distance_cache[b] = None

        geom_attn_bias_mask = None
        for _ in range(repeat_times):
            attn_mask_tensor = None
            if geom_attn_bias_mask is not None:
                B = geom_attn_bias_mask.size(0)
                geo_len = geom_attn_bias_mask.size(1)
                text_len = text_context.size(1)
                num_heads = self.text2geom_attn.num_heads
                bias = geom_attn_bias_mask.unsqueeze(1).unsqueeze(1)
                bias = bias.expand(B, num_heads, text_len, geo_len)
                attn_mask_tensor = bias.reshape(B * num_heads, text_len, geo_len)
            if (
                attn_mask_tensor is not None
                and geom_pad_mask.dtype != attn_mask_tensor.dtype
            ):
                geom_pad_mask = geom_pad_mask.to(dtype=attn_mask_tensor.dtype)
                if geom_pad_mask.is_floating_point():
                    geom_pad_mask = geom_pad_mask.masked_fill(
                        geom_pad_mask > 0, float("-inf")
                    )
            txt2geo_out, attn_txt2geo = self.text2geom_attn(
                query=text_context,
                key=geom_context,
                value=geom_context,
                key_padding_mask=geom_pad_mask,
                need_weights=True,
                attn_mask=attn_mask_tensor,
            )
            # attn_txt2geo: torch MHA  [B, T_q, T_k] (batch_first=True) [B, heads, T_q, T_k]
            if attn_txt2geo is not None:
                if attn_txt2geo.dim() == 4:
                    attn_map = attn_txt2geo.mean(dim=1)  # [B, T, G]
                else:
                    attn_map = attn_txt2geo  # [B, T, G]
                text2geom_attn_map = attn_map
                geom_importance = attn_map.mean(dim=1)  # [B, G]
                if self._capture_gradcam_llm:
                    self.last_text2geom_attn = attn_map.detach()
            else:
                geom_importance = None
                if self._capture_gradcam_llm:
                    self.last_text2geom_attn = None

            # Geom -> text cross-attention.
            geo2txt_out, _ = self.geom2text_attn(
                query=geom_context,
                key=text_context,
                value=text_context,
                key_padding_mask=text_pad_mask,
                need_weights=False,
                attn_mask=None,
            )

            # Residual updates.
            text_context = text_context + txt2geo_out
            geom_context = geom_context + geo2txt_out

        text_fused = text_context
        geom_fused = geom_context
        if geom_importance is not None:
            self.last_geom_importance = geom_importance.detach()
        else:
            self.last_geom_importance = None
        if geom_importance is not None:
            s_token = geom_importance / (geom_importance.sum(dim=1, keepdim=True) + 1e-8)
        geom_importance_bias = geom_importance
        self.last_distance_penalty = None

        geom_mask_bool = mask_bool.clone()
        geom_context_masked = geom_context * geom_mask_bool.unsqueeze(-1).to(dtype=geom_context.dtype)
        _, pre_attn_weights, _ = self.attention_pool(
            geom_context_masked,
            self.space_dict,
            temperature if temperature is not None else 0.0,
        )
        geom_mask_float = geom_mask_bool.to(dtype=pre_attn_weights.dtype)
        s_self_pre = s_self
        s_token_pre = s_token
        s_self_pre = s_self_pre * geom_mask_float
        s_token_pre = s_token_pre * geom_mask_float
        tis_pre = (s_self_pre + s_token_pre) / 2.0
        tis_pre = tis_pre / (tis_pre.sum(dim=1, keepdim=True) + 1e-8)
        self.last_tis_pre = tis_pre.detach()
        self.last_s_token_pre = s_token_pre.detach().cpu()
        self.last_token_dist = token_distance_cache

        mask = mask_bool
        # Reset per-step pruning trace for optional CIF export/debugging.
        self.last_geom_keep_indices = None
        self.last_geom_keep_mask = None
        orig_mask_bool = mask_bool.clone()
        pruning_active = True
        if self.geom_token_pruner is not None and geom_importance is not None:
            valid_lengths = geom_mask_bool.sum(dim=1).tolist()
            max_valid_len = max(1, int(max(valid_lengths)))
            compact_geom = geom_context.new_zeros(batch_size, max_valid_len, geom_context.size(-1))
            compact_bias = geom_importance_bias.new_zeros(batch_size, max_valid_len)
            compact_mandatory = None
            compact_tis = tis_pre.new_zeros(batch_size, max_valid_len)
            if mandatory_mask is not None:
                compact_mandatory = torch.zeros(batch_size, max_valid_len, device=geom_context.device, dtype=torch.bool)
            compact_valid_mask = torch.zeros(batch_size, max_valid_len, device=geom_context.device, dtype=torch.bool)
            max_text_len = max(1, int(text_mask_bool.sum(dim=1).max().item()))
            compact_attn = None
            compact_text_mask = None
            if text2geom_attn_map is not None:
                compact_attn = geom_importance.new_zeros(batch_size, max_text_len, max_valid_len)
                compact_text_mask = torch.zeros(batch_size, max_text_len, device=geom_context.device, dtype=torch.bool)
            for b in range(batch_size):
                valid_idx = torch.nonzero(geom_mask_bool[b], as_tuple=True)[0]
                if valid_idx.numel() == 0:
                    continue
                tokens = geom_context[b, valid_idx]
                compact_geom[b, :tokens.size(0)] = tokens
                compact_bias[b, :tokens.size(0)] = geom_importance_bias[b, valid_idx]
                compact_valid_mask[b, :tokens.size(0)] = True
                compact_tis[b, :tokens.size(0)] = tis_pre[b, valid_idx]
                if compact_mandatory is not None:
                    compact_mandatory[b, :tokens.size(0)] = mandatory_mask[b, valid_idx]
                if compact_attn is not None:
                    text_idx = torch.nonzero(text_mask_bool[b], as_tuple=True)[0]
                    if text_idx.numel() > 0:
                        attn_b = text2geom_attn_map[b]
                        compact_attn[b, :text_idx.size(0), :tokens.size(0)] = attn_b[text_idx][:, valid_idx]
                        compact_text_mask[b, :text_idx.size(0)] = True

            bias_vals = compact_bias
            pruned_geom, keep_indices_list, keep_lengths = self.geom_token_pruner.prune(
                compact_geom,
                space_dict=None,
                temperature=temperature if temperature is not None else 0.0,
                importance_bias=bias_vals,
                mandatory_keep_mask=compact_mandatory,
                valid_mask=compact_valid_mask,
                attention_scores=compact_attn,
                tis_scores=compact_tis,
                query_mask=compact_text_mask,
            )
            keep_mask_original = torch.zeros(
                batch_size,
                orig_mask_bool.size(1),
                device=geom_context.device,
                dtype=torch.bool,
            )
            keep_indices_original = []
            for b, keep_indices in enumerate(keep_indices_list):
                valid_idx = torch.nonzero(orig_mask_bool[b], as_tuple=True)[0]
                if valid_idx.numel() == 0 or keep_indices is None:
                    keep_indices_original.append(torch.empty(0, dtype=torch.long))
                    continue
                if not torch.is_tensor(keep_indices):
                    keep_indices = torch.as_tensor(keep_indices, device=valid_idx.device)
                keep_indices = keep_indices.to(device=valid_idx.device, dtype=torch.long)
                mapped = valid_idx[keep_indices]
                keep_mask_original[b, mapped] = True
                keep_indices_original.append(mapped.detach().cpu())
            self.last_geom_keep_indices = keep_indices_original
            self.last_geom_keep_mask = keep_mask_original.detach().cpu()
            geom_context = pruned_geom
            max_len = geom_context.size(1)
            geom_mask_bool = torch.zeros(batch_size, max_len, device=geom_context.device, dtype=torch.bool)
            for b, keep_len in enumerate(keep_lengths):
                if keep_len <= 0:
                    continue
                geom_mask_bool[b, :int(keep_len)] = True
            mask_bool = geom_mask_bool
            mask = geom_mask_bool

        geom_context_masked = geom_context * mask_bool.unsqueeze(-1).to(dtype=geom_context.dtype)
        llm_out = geom_context_masked
        if getattr(self.config, "capture_geom_importance", False):
            current_epoch = getattr(self, "_current_epoch", None)
            last_logged_epoch = getattr(self, "_geom_importance_logged_epoch", None)
            if current_epoch is None or current_epoch != last_logged_epoch:
                log_path = os.path.join(self.config.log_dir, "geom_importance.log")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                lines = []
                coords = x[..., :3].to(dtype=torch.float32)
                for b in range(batch_size):
                    valid_idx = torch.nonzero(mask_bool[b], as_tuple=True)[0]
                    if valid_idx.numel() == 0:
                        continue
                    imp_b = geom_importance[b, valid_idx]
                    topk = min(5, imp_b.numel())
                    top_vals, top_pos = torch.topk(imp_b, k=topk)
                    top_nodes = valid_idx[top_pos]
                    dist = None
                    if pos_mask is not None:
                        ads_idx = torch.nonzero(pos_mask[b], as_tuple=True)[0]
                        if ads_idx.numel() > 0:
                            dist = torch.cdist(
                                coords[b, top_nodes].unsqueeze(0),
                                coords[b, ads_idx].unsqueeze(0),
                                p=2,
                            ).squeeze(0).min(dim=1).values
                    dist_cpu = dist.detach().cpu().tolist() if dist is not None else None
                    lines.append(
                        f"[GeomImportance] sample={b} top_nodes={top_nodes.detach().cpu().tolist()} "
                        f"importance={top_vals.detach().cpu().tolist()} dist_to_ads={dist_cpu}"
                    )
                if lines:
                    with open(log_path, "a", encoding="utf-8") as f:
                        for ln in lines:
                            f.write(ln + "\n")
                    if current_epoch is not None:
                        self._geom_importance_logged_epoch = current_epoch

        self.last_distance_penalty = None

        cls_feature, attn_weight_flat, pruned_feature = self.attention_pool(llm_out, self.space_dict, temperature)
        geom_len = geom_mask_bool.size(1)
        geom_true_counts = geom_mask_bool.sum(dim=1).to(dtype=torch.long)
        s_cls_raw = torch.zeros(attn_weight_flat.size(0), geom_len, device=attn_weight_flat.device, dtype=attn_weight_flat.dtype)
        total_tokens = attn_weight_flat.size(1)
        for b in range(attn_weight_flat.size(0)):
            true_count = int(geom_true_counts[b].item())
            if true_count <= 0:
                continue
            take = min(true_count, total_tokens)
            if take <= 0:
                continue
            geom_scores = attn_weight_flat[b, :take]
            true_positions = torch.nonzero(geom_mask_bool[b], as_tuple=True)[0]
            if true_positions.numel() < take:
                geom_scores = geom_scores[: true_positions.numel()]
                take = geom_scores.size(0)
            s_cls_raw[b, true_positions[:take]] = geom_scores
        s_cls = s_cls_raw / (s_cls_raw.sum(dim=1, keepdim=True) + 1e-8)

        def _align_score(score):
            if score is None:
                return None
            if score.size(1) > geom_len:
                return score[:, :geom_len]
            if score.size(1) < geom_len:
                pad = torch.zeros(score.size(0), geom_len - score.size(1), device=score.device, dtype=score.dtype)
                return torch.cat([score, pad], dim=1)
            return score

        s_self = _align_score(s_self)
        s_token = _align_score(s_token)
        if s_self is None:
            s_self = s_cls.clone()
        if s_token is None:
            s_token = s_cls.clone()
        s_self = s_self / (s_self.sum(dim=1, keepdim=True) + 1e-8)
        s_token = s_token / (s_token.sum(dim=1, keepdim=True) + 1e-8)
        tis = (s_cls + s_self + s_token) / 3.0
        self.last_s_cls = s_cls.detach()
        self.last_s_self = s_self.detach()
        self.last_s_token = s_token.detach()
        self.last_tis = tis.detach()

        node_energy = self.geom_energy_head(geom_context).squeeze(-1)
        node_energy = node_energy * mask_bool.to(dtype=node_energy.dtype)
        summed = node_energy.sum(dim=1)
        motif_nodes = mask_bool.sum(dim=1).clamp(min=1).to(dtype=node_energy.dtype)
        normalized_pred = summed / motif_nodes
        adsorption_energy = normalized_pred * self.target_std + self.target_mean
        loss_pruning = 0.0

        if return_feature:
            return adsorption_energy, cls_feature, loss_pruning
        return adsorption_energy, loss_pruning

