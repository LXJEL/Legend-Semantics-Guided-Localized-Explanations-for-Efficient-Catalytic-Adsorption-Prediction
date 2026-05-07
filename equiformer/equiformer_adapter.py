import os
import sys
from types import SimpleNamespace

import torch
from torch import nn


def _resolve_equiformer_root(config_root=None):
    if config_root:
        return config_root
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # ../../equiformer_v2-main/equiformer_v2-main relative to this file
    candidate = os.path.normpath(
        os.path.join(base_dir, "..", "..", "equiformer_v2-main", "equiformer_v2-main")
    )
    return candidate


class EquiformerV2TokenEncoder(nn.Module):
    """
    Wrap EquiformerV2 to return per-atom scalar features (l=0).
    Uses provided edge_index instead of OTF graph construction.
    """

    def __init__(self, equiformer_root, hidden_nf, max_num_elements=100, max_radius=5.0, **kwargs):
        super().__init__()
        equiformer_root = _resolve_equiformer_root(equiformer_root)
        if not os.path.isdir(equiformer_root):
            raise FileNotFoundError(f"EquiformerV2 root not found: {equiformer_root}")
        if equiformer_root not in sys.path:
            sys.path.append(equiformer_root)

        from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
        from nets.equiformer_v2.so3 import SO3_Embedding

        ckpt_path = kwargs.pop("ckpt_path", None)
        max_neighbors = int(kwargs.pop("max_neighbors", 20))
        self.model = EquiformerV2_OC20(
            num_atoms=None,
            bond_feat_dim=None,
            num_targets=None,
            use_pbc=True,
            regress_forces=False,
            otf_graph=True,
            max_neighbors=max_neighbors,
            max_radius=float(max_radius),
            max_num_elements=max_num_elements,
            sphere_channels=hidden_nf,
            **kwargs,
        )
        if ckpt_path:
            state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state, dict):
                for key in ("state_dict", "model_state_dict", "model"):
                    if key in state:
                        state = state[key]
                        break
            if not isinstance(state, dict):
                raise ValueError("Equiformer checkpoint does not contain a state_dict.")
            model_keys = set(self.model.state_dict().keys())
            new_state = {}
            for k, v in state.items():
                if k.startswith("module."):
                    k = k[len("module."):]
                # Try direct, model., and encoder.model. variants to match this build.
                if k in model_keys:
                    new_state[k] = v
                    continue
                if f"model.{k}" in model_keys:
                    new_state[f"model.{k}"] = v
                    continue
                if f"encoder.model.{k}" in model_keys:
                    new_state[f"encoder.model.{k}"] = v
                    continue
                # If ckpt already has model./encoder.model., keep if it matches.
                if k.startswith("model.") and k in model_keys:
                    new_state[k] = v
                    continue
                if k.startswith("encoder.model.") and k in model_keys:
                    new_state[k] = v
                    continue
            missing, unexpected = self.model.load_state_dict(new_state, strict=True)
            if missing:
                print(f"[EquiformerV2Adapter] missing keys: {len(missing)}")
            if unexpected:
                print(f"[EquiformerV2Adapter] unexpected keys: {len(unexpected)}")
        self._so3_embedding = SO3_Embedding
        self.hidden_nf = hidden_nf

    def _generate_graph(self, data):
        edge_index = data.edge_index
        pos = data.pos
        if edge_index is None or edge_index.numel() == 0:
            empty = pos.new_empty((0,), dtype=pos.dtype)
            empty_vec = pos.new_empty((0, 3), dtype=pos.dtype)
            cell_offsets = pos.new_empty((0, 3), dtype=pos.dtype)
            return edge_index, empty, empty_vec, cell_offsets, None, None
        edge_offsets = getattr(data, "edge_offsets", None)
        if edge_offsets is not None and edge_offsets.numel() > 0:
            edge_vec = pos[edge_index[0]] - pos[edge_index[1]] + edge_offsets
            cell_offsets = edge_offsets
        else:
            edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
            cell_offsets = pos.new_zeros((edge_index.size(1), 3))
        edge_dist = torch.linalg.norm(edge_vec, dim=1).clamp_min(1e-12)
        return edge_index, edge_dist, edge_vec, cell_offsets, None, None

    def forward_features(self, data):
        atomic_numbers = data.atomic_numbers.long()
        num_atoms = atomic_numbers.numel()
        if num_atoms == 0:
            return data.pos.new_zeros((0, self.hidden_nf))

        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            _cell_offsets,
            _,
            _neighbors,
        ) = self._generate_graph(data)

        if edge_index is None or edge_index.numel() == 0:
            return data.pos.new_zeros((num_atoms, self.hidden_nf))

        edge_rot_mat = self.model._init_edge_rot_mat(data, edge_index, edge_distance_vec)
        for i in range(self.model.num_resolutions):
            self.model.SO3_rotation[i].set_wigner(edge_rot_mat)

        x = self._so3_embedding(
            num_atoms,
            self.model.lmax_list,
            self.model.sphere_channels,
            data.pos.device,
            data.pos.dtype,
        )

        offset_res = 0
        offset = 0
        for i in range(self.model.num_resolutions):
            if self.model.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.model.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.model.sphere_embedding(
                    atomic_numbers
                )[:, offset : offset + self.model.sphere_channels]
            offset = offset + self.model.sphere_channels
            offset_res = offset_res + int((self.model.lmax_list[i] + 1) ** 2)

        edge_distance = self.model.distance_expansion(edge_distance)
        if self.model.share_atom_edge_embedding and self.model.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]
            target_element = atomic_numbers[edge_index[1]]
            source_embedding = self.model.source_embedding(source_element)
            target_embedding = self.model.target_embedding(target_element)
            edge_distance = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)

        edge_degree = self.model.edge_degree_embedding(
            atomic_numbers,
            edge_distance,
            edge_index,
        )
        x.embedding = x.embedding + edge_degree.embedding

        for i in range(self.model.num_layers):
            x = self.model.blocks[i](
                x,
                atomic_numbers,
                edge_distance,
                edge_index,
                batch=data.batch,
            )

        x.embedding = self.model.norm(x.embedding)

        # Return l=0 scalar features: [num_atoms, hidden_nf]
        return x.embedding[:, 0, :].contiguous()


class EquiformerV2Adapter(nn.Module):
    """
    Adapter that matches the EGNN forward signature and returns padded per-atom tokens.
    """

    def __init__(self, equiformer_root, hidden_nf, max_len, max_num_elements=100, max_radius=5.0, **kwargs):
        super().__init__()
        self.encoder = EquiformerV2TokenEncoder(
            equiformer_root,
            hidden_nf=hidden_nf,
            max_num_elements=max_num_elements,
            max_radius=max_radius,
            **kwargs,
        )
        self.hidden_nf = hidden_nf
        self.max_len = max_len

    def forward(self, h, x, edges, mask, batch_size, pos_mask=None):
        if h.dim() == 3 and h.size(-1) == 1:
            h = h.squeeze(-1)
        z = h.to(dtype=torch.long)
        coords = x[..., :3].to(dtype=torch.float32)
        mask_bool = mask.to(dtype=torch.bool)

        device = coords.device
        z_list = []
        pos_list = []
        batch_list = []
        edge_list = []
        edge_offsets_list = []
        natoms = []
        offset = 0

        if not isinstance(edges, (list, tuple)):
            edges = [edges for _ in range(batch_size)]

        for b in range(batch_size):
            valid_idx = torch.nonzero(mask_bool[b], as_tuple=True)[0]
            n = int(valid_idx.numel())
            natoms.append(n)
            if n == 0:
                continue
            z_list.append(z[b, valid_idx])
            pos_list.append(coords[b, valid_idx])
            batch_list.append(torch.full((n,), b, device=device, dtype=torch.long))

            edge_entry = edges[b]
            edge_offsets = None
            if isinstance(edge_entry, (list, tuple)) and len(edge_entry) == 2:
                edge_idx, edge_offsets = edge_entry
            else:
                edge_idx = edge_entry
            if edge_idx is not None and not torch.is_tensor(edge_idx):
                edge_idx = torch.tensor(edge_idx, device=device, dtype=torch.long)
            if edge_idx is not None and edge_idx.numel() > 0:
                edge_list.append(edge_idx.to(device=device, dtype=torch.long) + offset)
                if edge_offsets is not None:
                    edge_offsets_list.append(edge_offsets.to(device=device, dtype=coords.dtype))
            offset += n

        total_atoms = sum(natoms)
        out = coords.new_zeros((batch_size, mask.size(1), self.hidden_nf))
        if total_atoms == 0:
            return out

        z_all = torch.cat(z_list, dim=0)
        pos_all = torch.cat(pos_list, dim=0)
        batch_all = torch.cat(batch_list, dim=0)
        if edge_list:
            edge_index = torch.cat(edge_list, dim=1)
            if edge_offsets_list and len(edge_offsets_list) == len(edge_list):
                edge_offsets = torch.cat(edge_offsets_list, dim=0)
            else:
                edge_offsets = None
        else:
            edge_index = torch.empty((2, 0), device=device, dtype=torch.long)
            edge_offsets = None

        data = SimpleNamespace(
            atomic_numbers=z_all,
            pos=pos_all,
            batch=batch_all,
            natoms=natoms,
            edge_index=edge_index,
            edge_offsets=edge_offsets,
        )
        node_feats = self.encoder.forward_features(data)

        offset = 0
        for b, n in enumerate(natoms):
            if n <= 0:
                continue
            out[b, :n] = node_feats[offset : offset + n]
            offset += n
        return out
