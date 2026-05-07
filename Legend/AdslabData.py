import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers.tokenization_utils_base import BatchEncoding
from torch.utils.data import Dataset, DataLoader, get_worker_info
from torch.utils.data.distributed import DistributedSampler

from ase.data import atomic_numbers as ASE_ATOMIC_NUMBERS

try:
    from ocpmodels.common.utils import radius_graph_pbc
except Exception:  # pragma: no cover - optional dependency
    radius_graph_pbc = None


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class AdslabData(Dataset):
    """
    Dataset wrapper that keeps the parent process lightweight.
    Large tensors are loaded lazily per worker to avoid massive
    shared-memory allocations when num_workers > 0.
    """

    _cached_fields = [
        "strings",
        "elements",
        "connectivities",
        "positions",
        "pos_mask",
        "targets",
        "sids",
        "cells",
        "pbcs",
    ]

    def __init__(
        self,
        path,
        text_max_len,
        device,
        edge_method="full",
        edge_cutoff=4.5,
        edge_knn=16,
        ads_cutoff_scale=1.5,
        ads_knn_scale=1.5,
        normalize_labels=False,
    ):
        super().__init__()
        self.device = device
        self.path = path
        self.text_max_len = text_max_len
        self.perm = None
        self._data_loaded = False
        self.edge_method = (edge_method or "full").lower()
        if self.edge_method not in {"full", "hybrid"}:
            self.edge_method = "full"
        self.edge_cutoff = float(edge_cutoff)
        self.edge_knn = max(0, int(edge_knn))
        self.ads_cutoff_scale = float(ads_cutoff_scale)
        self.ads_knn_scale = float(ads_knn_scale)
        self.normalize_labels = bool(normalize_labels)

        # Placeholder attributes for lazy materialization
        for field in self._cached_fields:
            setattr(self, field, None)

        # Load metadata once (length + stats), then drop references.
        temp_data = torch.load(path)
        if len(temp_data) < 7:
            raise ValueError(f"{path}7{len(temp_data)}")
        temp_strings, temp_elements, temp_connectivities, temp_positions, temp_pos_mask, temp_targets, temp_sids = temp_data[:7]
        lengths = [
            len(temp_strings), len(temp_elements), len(temp_positions),
            len(temp_targets), len(temp_connectivities), len(temp_pos_mask), len(temp_sids)
        ]
        if len(temp_data) >= 9:
            temp_cells, temp_pbcs = temp_data[7:9]
            lengths.extend([len(temp_cells), len(temp_pbcs)])
        if len(set(lengths)) != 1:
            raise ValueError(f": {lengths}")
        self._length = lengths[0]

        self.is_train = "train" in Path(path).name.lower()
        self.stats_path = os.path.join(Path(path).parent, "target_stats.pt")
        if self.is_train:
            if self.normalize_labels:
                all_targets = torch.tensor(temp_targets, dtype=torch.float32)
                self.target_mean = all_targets.mean().item()
                self.target_std = all_targets.std().item()
            else:
                self.target_mean = 0.0
                self.target_std = 1.0
            torch.save({"mean": self.target_mean, "std": self.target_std}, self.stats_path)
        else:
            if os.path.exists(self.stats_path):
                stats = torch.load(self.stats_path)
                self.target_mean = stats["mean"]
                self.target_std = stats["std"]
            else:
                self.target_mean = 0.0
                self.target_std = 1.0

        # Release temporary references to keep the pickled dataset lightweight
        del temp_strings, temp_elements, temp_connectivities, temp_positions, temp_pos_mask, temp_targets, temp_sids, temp_data

    def __len__(self):
        return self._length

    # ------------------------------------------------------------------ #
    # Lazy loading helpers
    # ------------------------------------------------------------------ #
    def _load_data(self):
        if self._data_loaded:
            return
        data = torch.load(self.path)
        if len(data) >= 9:
            fields = data[:9]
        else:
            fields = list(data[:7]) + [[None] * self._length, [None] * self._length]
        for field, value in zip(self._cached_fields, fields):
            setattr(self, field, value)
        self._data_loaded = True

    def _ensure_loaded(self):
        if not self._data_loaded:
            self._load_data()

    def __getstate__(self):
        state = self.__dict__.copy()
        if state.get("_data_loaded"):
            for field in self._cached_fields:
                state[field] = None
            state["_data_loaded"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "_data_loaded"):
            self._data_loaded = False
        for field in self._cached_fields:
            if not hasattr(self, field):
                setattr(self, field, None)

    # ------------------------------------------------------------------ #
    # Dataset interface
    # ------------------------------------------------------------------ #
    def __getitem__(self, index):
        self._ensure_loaded()
        original_index = index
        if self.perm is not None:
            index = self.perm[index]
        try:
            string = self.strings[index]
            element = self.elements[index]
            position = self.positions[index]              # torch.Tensor [N,3]
            target = self.targets[index]                  # float (per-atom energy)
            connectivity = self.connectivities[index]     # np.ndarray [N,N] or torch.Tensor
            pos_mask = self.pos_mask[index]               # torch.BoolTensor [N]
            sid = self.sids[index]
            cell = self.cells[index] if self.cells is not None else None
            pbc = self.pbcs[index] if self.pbcs is not None else None

            natoms = len(element)
            if natoms > self.text_max_len:
                raise RuntimeError(f"sample {original_index}: natoms {natoms} > text_max_len {self.text_max_len}")

            atomic_nums = [ASE_ATOMIC_NUMBERS.get(atom, 0) for atom in element]
            h = torch.tensor(atomic_nums, dtype=torch.long)

            position_tensor = position.clone().detach().to(dtype=torch.float)
            x = position_tensor

            if isinstance(connectivity, np.ndarray):
                conn = torch.from_numpy(connectivity)
            else:
                conn = connectivity
            conn = conn.to(dtype=torch.float)

            pos_mask_core = pos_mask.clone().detach().to(dtype=torch.bool)
            edges = self._build_edge_index(
                position_tensor[:natoms],
                conn,
                pos_mask_core[:natoms],
                cell=cell,
                pbc=pbc,
            )

            normalized_target = (target - self.target_mean) / (self.target_std + 1e-8)
            if not self.normalize_labels:
                normalized_target = target
            target_tensor = torch.tensor(normalized_target, dtype=torch.float).view(-1)  # [1]
            raw_target_tensor = torch.tensor(float(target), dtype=torch.float).view(-1)  # [1]

            pos_mask = pos_mask.clone().detach().to(dtype=torch.bool)
            mask = torch.ones(natoms, dtype=torch.bool)

            sidt = torch.zeros(1, dtype=torch.float)
            # Use original dataset index so CIF ids align with val.pt ordering.
            sidt[0] = float(original_index)
            sid_str = str(sid)

            return [h, x, edges, target_tensor, raw_target_tensor, string, mask, pos_mask, sidt, sid_str, cell, pbc]

        except Exception as e:
            raise RuntimeError(f"sample {original_index} failed: {str(e)}") from e

    def _build_edge_index(self, coords, connectivity, ads_mask, cell=None, pbc=None):
        """Construct edges to match EquiformerV2 (radius_graph_pbc only)."""
        natoms = coords.shape[0]
        if natoms <= 1:
            empty_idx = torch.empty((2, 0), dtype=torch.long)
            empty_off = torch.empty((0, 3), dtype=coords.dtype)
            return empty_idx, empty_off

        if radius_graph_pbc is None:
            raise RuntimeError("radius_graph_pbc is required for v2-style PBC graphs.")
        if cell is None or pbc is None:
            raise RuntimeError("PBC cell/pbc required for v2-style graphs.")

        coords = coords.to(dtype=torch.float)
        cell_t = torch.as_tensor(cell, dtype=coords.dtype, device=coords.device).view(1, 3, 3)
        pbc_t = torch.as_tensor(pbc, dtype=torch.bool, device=coords.device)
        if not torch.any(pbc_t):
            raise RuntimeError("PBC is disabled in data; v2-style graphs require use_pbc=True.")

        data = SimpleNamespace(
            pos=coords,
            natoms=torch.tensor([coords.size(0)], device=coords.device, dtype=torch.long),
            cell=cell_t,
        )
        edge_index, cell_offsets, _ = radius_graph_pbc(
            data, float(self.edge_cutoff), int(self.edge_knn)
        )
        if edge_index is None or edge_index.numel() == 0:
            empty_idx = torch.empty((2, 0), dtype=torch.long)
            empty_off = torch.empty((0, 3), dtype=coords.dtype)
            return empty_idx, empty_off
        offsets = cell_offsets.to(dtype=coords.dtype, device=coords.device) @ cell_t[0]
        return edge_index, offsets

    def _apply_knn(self, dist, ads_mask):
        """Return adjacency mask from k-nearest neighbors."""
        natoms = dist.size(0)
        adjacency = torch.zeros((natoms, natoms), dtype=torch.bool)
        if natoms <= 1:
            return adjacency

        base_k = min(self.edge_knn, natoms - 1)
        if base_k <= 0:
            return adjacency
        node_k = torch.full((natoms,), base_k, dtype=torch.long)
        if ads_mask.any():
            ads_indices = ads_mask.nonzero(as_tuple=False).view(-1)
            scaled_k = int(round(base_k * max(self.ads_knn_scale, 1.0)))
            scaled_k = max(1, min(scaled_k, natoms - 1))
            node_k[ads_indices] = scaled_k

        for i in range(natoms):
            k = int(node_k[i].item())
            if k <= 0:
                continue
            k = min(k, natoms - 1)
            vals, idx = torch.topk(dist[i], k, largest=False)
            adjacency[i, idx] = True
        return adjacency

    def _apply_radius_neighbors(self, dist, node_cutoffs, ads_mask):
        natoms = dist.size(0)
        adjacency = torch.zeros((natoms, natoms), dtype=torch.bool, device=dist.device)
        max_neighbors = max(0, int(self.edge_knn))
        node_k = None
        if ads_mask.any() and max_neighbors > 0 and self.ads_knn_scale != 1.0:
            node_k = torch.full((natoms,), max_neighbors, dtype=torch.long, device=dist.device)
            ads_indices = ads_mask.nonzero(as_tuple=False).view(-1)
            scaled_k = int(round(max_neighbors * max(self.ads_knn_scale, 1.0)))
            scaled_k = max(1, min(scaled_k, natoms - 1))
            node_k[ads_indices] = scaled_k

        for i in range(natoms):
            cutoff = float(node_cutoffs[i].item())
            if cutoff <= 0:
                continue
            within = dist[i] <= cutoff
            within[i] = False
            idx = torch.nonzero(within, as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            if max_neighbors > 0:
                k = max_neighbors if node_k is None else int(node_k[i].item())
                if idx.numel() > k:
                    vals = dist[i][idx]
                    _, order = torch.topk(vals, k, largest=False)
                    idx = idx[order]
            adjacency[i, idx] = True
        return adjacency

    def _pairwise_distance(self, coords, cell=None, pbc=None, return_shifts=False):
        if cell is None or pbc is None:
            dist = torch.cdist(coords, coords, p=2)
            return (dist, None) if return_shifts else dist
        cell_t = torch.as_tensor(cell, dtype=coords.dtype, device=coords.device)
        pbc_t = torch.as_tensor(pbc, dtype=torch.bool, device=coords.device)
        if cell_t.numel() != 9 or pbc_t.numel() != 3 or not torch.any(pbc_t):
            dist = torch.cdist(coords, coords, p=2)
            return (dist, None) if return_shifts else dist
        try:
            inv_cell = torch.inverse(cell_t)
        except RuntimeError:
            dist = torch.cdist(coords, coords, p=2)
            return (dist, None) if return_shifts else dist
        diff = coords[:, None, :] - coords[None, :, :]
        frac = torch.matmul(diff, inv_cell.t())
        pbc_mask = pbc_t.to(dtype=coords.dtype).view(1, 1, 3)
        shift = torch.round(frac) * pbc_mask
        frac = frac - shift
        diff = torch.matmul(frac, cell_t)
        dist = torch.norm(diff, dim=-1)
        if return_shifts:
            return dist, shift
        return dist

    def shuffle(self):
        self.perm = torch.randperm(len(self)).tolist()
        return self


class TrainCollater(object):
    def __init__(self, tokenizer, text_max_len, device):
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        self.device = device

    def __call__(self, batch):
        try:
            if len(batch[0]) >= 12:
                h_list, x_list, edges_list, target_list, raw_target_list, text_list, mask_list, pos_mask_list, sid_list, sid_str_list, cell_list, pbc_list = zip(*batch)
            else:
                h_list, x_list, edges_list, target_list, raw_target_list, text_list, mask_list, pos_mask_list, sid_list, cell_list, pbc_list = zip(*batch)
                sid_str_list = None

            max_natoms = max(h.size(0) for h in h_list)
            device = h_list[0].device
            h = torch.zeros(len(h_list), max_natoms, dtype=torch.long, device=device)
            x = torch.zeros(len(x_list), max_natoms, 3, dtype=torch.float, device=device)
            mask = torch.zeros(len(mask_list), max_natoms, dtype=torch.bool, device=device)
            pos_mask = torch.zeros(len(pos_mask_list), max_natoms, dtype=torch.bool, device=device)
            for i, (h_i, x_i, m_i, pm_i) in enumerate(zip(h_list, x_list, mask_list, pos_mask_list)):
                n = h_i.size(0)
                h[i, :n] = h_i
                x[i, :n] = x_i
                mask[i, :n] = m_i
                pos_mask[i, :n] = pm_i
            edges = []
            for edge in edges_list:
                if isinstance(edge, (list, tuple)) and len(edge) == 2:
                    edge_idx, edge_offsets = edge
                    edge_idx = edge_idx.clone() if hasattr(edge_idx, "clone") else edge_idx
                    if edge_offsets is not None and hasattr(edge_offsets, "clone"):
                        edge_offsets = edge_offsets.clone()
                    edges.append((edge_idx, edge_offsets))
                else:
                    edges.append(edge.clone())
            target = torch.stack(target_list, dim=0)
            raw_target = torch.stack(raw_target_list, dim=0)
            sid = torch.stack(sid_list, dim=0)

            text_batch = self.tokenizer(
                text_list,
                padding='longest',
                truncation=True,
                max_length=self.text_max_len,
                return_tensors='pt'
            )

            if sid_str_list is not None:
                return [h, x, edges, target, raw_target, text_batch, mask, pos_mask, sid, list(sid_str_list), list(cell_list), list(pbc_list)]
            return [h, x, edges, target, raw_target, text_batch, mask, pos_mask, sid, list(cell_list), list(pbc_list)]

        except Exception as e:
            raise RuntimeError(f"collater unpack failed: {str(e)}") from e


class DataPreFetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.next_data = self._move_to_cuda(self.next_data)

    def _move_to_cuda(self, data):
        if isinstance(data, torch.Tensor):
            return data if data.is_cuda else data.cuda(non_blocking=True)
        elif isinstance(data, BatchEncoding):
            return BatchEncoding({k: self._move_to_cuda(v) for k, v in data.items()})
        elif isinstance(data, dict):
            return {k: self._move_to_cuda(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._move_to_cuda(x) for x in data]
        else:
            return data

    def next(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        if data is not None:
            self.preload()
        return data


class AdslabDataloader:
    def __init__(
        self,
        batch_size: int = 32,
        root: str = 'oc20data_proc',
        text_max_len: int = 512,
        device: str = 'cuda',
        tokenizer=None,
        train_file: str = 'train_pa.pt',
        val_file: str = 'val_id_pa.pt',
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
        edge_method: str = "full",
        edge_cutoff: float = 4.5,
        edge_knn: int = 16,
        ads_cutoff_scale: float = 1.5,
        ads_knn_scale: float = 1.5,
        normalize_labels: bool = False,
    ):
        self.batch_size = batch_size
        self.text_max_len = text_max_len
        if tokenizer is None:
            raise ImportError(
                "A tokenizer must be provided. This supplement only ships the paper method."
            )
        self.tokenizer = tokenizer
        self.device = device
        self.num_workers = getattr(self.tokenizer, "_num_workers", None)
        self.root = root
        self.train_file = train_file
        self.val_file = val_file
        self.edge_method = edge_method
        self.edge_cutoff = edge_cutoff
        self.edge_knn = edge_knn
        self.ads_cutoff_scale = ads_cutoff_scale
        self.ads_knn_scale = ads_knn_scale
        self.normalize_labels = bool(normalize_labels)

        self.distributed = distributed
        self.world_size = max(1, world_size)
        self.rank = rank

        self.train_dataset = AdslabData(
            os.path.join(root, train_file),
            text_max_len=self.text_max_len,
            device=self.device,
            edge_method=self.edge_method,
            edge_cutoff=self.edge_cutoff,
            edge_knn=self.edge_knn,
            ads_cutoff_scale=self.ads_cutoff_scale,
            ads_knn_scale=self.ads_knn_scale,
            normalize_labels=getattr(self, "normalize_labels", False),
        )
        self.val_dataset = AdslabData(
            os.path.join(root, val_file),
            text_max_len=self.text_max_len,
            device=self.device,
            edge_method=self.edge_method,
            edge_cutoff=self.edge_cutoff,
            edge_knn=self.edge_knn,
            ads_cutoff_scale=self.ads_cutoff_scale,
            ads_knn_scale=self.ads_knn_scale,
            normalize_labels=getattr(self, "normalize_labels", False),
        )

    def _loader_kwargs(self, dataset, shuffle, use_distributed_sampler):
        num_workers = self.num_workers if self.num_workers is not None else 0
        sampler = None
        if use_distributed_sampler and self.distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
            )
            shuffle = False
        kwargs = dict(
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.device),
        )
        if num_workers > 0:
            kwargs["prefetch_factor"] = 2
            kwargs["persistent_workers"] = True
            kwargs["worker_init_fn"] = seed_worker
        return kwargs

    def train_dataloader(self):
        loader_kwargs = self._loader_kwargs(self.train_dataset, shuffle=True, use_distributed_sampler=True)
        return DataLoader(self.train_dataset, **loader_kwargs)

    def val_dataloader(self):
        loader_kwargs = self._loader_kwargs(self.val_dataset, shuffle=False, use_distributed_sampler=False)
        return DataLoader(self.val_dataset, **loader_kwargs)
