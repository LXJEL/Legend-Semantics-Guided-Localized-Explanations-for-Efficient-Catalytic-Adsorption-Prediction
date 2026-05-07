# -*- coding: utf-8 -*-
"""
Utility script to preprocess OC20 IS2RE data into the CatBERTa training
tuple saved as a .pt file.

Examples:

IS2RE
python preprocess_data.py \
  --lmdb_path "/path/to/oc20/is2re/train_200k.lmdb" \
  --mapping_path "/path/to/oc20_data_mapping.pkl" \
  --save_path "/path/to/output/train.pt"
"""

import argparse
import os
import pickle
import csv
from collections import Counter
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import tqdm
from ase import Atoms, cell, constraints
from ocpmodels.datasets import LmdbDataset

from site_analysis import SiteAnalyzer


def sanitize_ads_symbol(symbol: str) -> str:
    """Remove CatBERT-specific artefacts."""
    return symbol.replace("*", "").replace(" ", "")


class OC20DataGenerator:
    def __init__(
        self,
        lmdb_path: str,
        mapping_path: str,
        save_path: str,
        max_text_len: int = 512,
        limit: Optional[int] = None,
        seed: Optional[int] = None,
        debug: bool = False,
        debug_print_limit: int = 5,
        split_names: Optional[Sequence[str]] = None,
        split_save_dir: Optional[str] = None,
        split_sample_limit: Optional[int] = None,
        split_summary_path: Optional[str] = None,
    ):
        self.lmdb_data = LmdbDataset({"src": lmdb_path})
        self.mapping: Dict[str, Dict[str, Any]] = pickle.load(open(mapping_path, "rb"))
        self.save_path = save_path
        self.max_text_len = max_text_len
        self.limit = limit
        self.seed = seed
        self.use_relaxed_positions = True

        self.debug = debug
        self.debug_print_limit = max(int(debug_print_limit), 0)

        self.output_dir = os.path.dirname(save_path) or "."
        os.makedirs(self.output_dir, exist_ok=True)
        self.metadata_energy_map: Dict[str, Optional[float]] = {}
        self.energy_mean = 0.0
        self.energy_std = 0.0
        self._prepare_metadata_energy_stats()
        if not split_names:
            split_names = ["train"]
        clean_splits = [str(name).strip().lower() for name in split_names if str(name).strip()]
        if not clean_splits:
            clean_splits = ["train"]
        self.target_splits = tuple(clean_splits)
        self.split_sample_limit = split_sample_limit
        base_save_dir = split_save_dir or (os.path.dirname(save_path) if save_path and len(self.target_splits) > 1 else self.output_dir)
        os.makedirs(base_save_dir, exist_ok=True)
        if len(self.target_splits) == 1 and save_path:
            self.split_save_paths = {self.target_splits[0]: save_path}
        else:
            self.split_save_paths = {
                split: os.path.join(base_save_dir, f"{split}_pa.pt") for split in self.target_splits
            }
        if split_summary_path:
            self.split_summary_path = split_summary_path
        else:
            default_summary_dir = os.path.dirname(save_path) if save_path else self.output_dir
            self.split_summary_path = os.path.join(default_summary_dir, "split_summary.csv")

    @staticmethod
    def _ensure_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        return np.asarray(value)

    @staticmethod
    def _normalize_sid(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().view(-1)[0].item()
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _extract_metadata_energy(record: Dict[str, Any]) -> Optional[float]:
        energy_keys = [
            "adsorption_energy",
            "adsorption_energy_ev",
            "adsorption_energy_total",
            "ea_ev",
            "adsorption_energy_ev_ml",
            "energy",
        ]
        for key in energy_keys:
            if key in record and record[key] is not None:
                try:
                    return float(record[key])
                except (TypeError, ValueError):
                    continue
        return None

    def _prepare_metadata_energy_stats(self):
        energies = []
        energy_map = {}
        for sid, record in self.mapping.items():
            norm_sid = str(self._normalize_sid(sid))
            energy = self._extract_metadata_energy(record)
            energy_map[norm_sid] = energy
            if energy is not None:
                energies.append(float(energy))
        self.metadata_energy_map = energy_map
        if energies:
            arr = np.asarray(energies, dtype=np.float64)
            self.energy_mean = float(arr.mean())
            self.energy_std = float(arr.std()) if arr.size > 1 else 0.0
        else:
            self.energy_mean = 0.0
            self.energy_std = 0.0

    def _energy_delta(self, sid: str, candidate_energy: float) -> float:
        canonical = self.metadata_energy_map.get(str(sid))
        if canonical is None:
            return abs(candidate_energy)
        return abs(candidate_energy - canonical)

    def _write_log_file(self, filename: str, lines):
        if not lines:
            return
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        print(f"[LOG] Saved {filename} with {len(lines)} entries.")

    def _write_frequency_csv(self, ads_labels, bulk_labels, miller_labels, suffix: Optional[str] = None):
        if suffix:
            filename = f"frequency_stats_{suffix}.csv"
        else:
            filename = "frequency_stats.csv"
        path = os.path.join(self.output_dir, filename)
        counter_specs = [
            ("adsorbate", Counter(ads_labels)),
            ("bulk", Counter(bulk_labels)),
            ("miller", Counter(str(m) for m in miller_labels)),
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["type", "name", "count"])
            for label_type, counter in counter_specs:
                for name, count in counter.most_common():
                    writer.writerow([label_type, name, count])
        print(f"[INFO] Frequency CSV written to {path}")

    def _write_split_summary(self, raw_counts: Counter, kept_counts: Counter):
        if not raw_counts and not kept_counts:
            return
        path = self.split_summary_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        all_keys = sorted(set(raw_counts.keys()) | set(kept_counts.keys()))
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "raw_count", "kept_count"])
            for key in all_keys:
                writer.writerow([key, raw_counts.get(key, 0), kept_counts.get(key, 0)])
        print(f"[INFO] Split summary written to {path}")

    def _init_bucket(self):
        return {
            "strings": [],
            "elements": [],
            "connectivities": [],
            "positions": [],
            "pos_mask": [],
            "labels": [],
            "sids": [],
            "cells": [],
            "pbcs": [],
            "ads_labels": [],
            "bulk_labels": [],
            "miller_labels": [],
            "seen_sids": {},
            "duplicate_log": [],
            "outlier_log": [],
        }

    @staticmethod
    def _to_float(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                raise ValueError("Empty tensor provided when extracting scalar energy.")
            return float(value.detach().cpu().view(-1)[0].item())
        return float(value)

    def _extract_energy(self, pyg) -> float:
        candidate_attrs = ("y_relaxed", "y", "energy", "total_energy")
        for attr in candidate_attrs:
            if not hasattr(pyg, attr):
                continue
            value = getattr(pyg, attr)
            if value is None:
                continue
            return self._to_float(value)
        raise ValueError("No energy attribute found on sample; expected one of y_relaxed/y/energy/total_energy.")

    def _atoms_from_pyg(self, pyg, relaxed: bool) -> Atoms:
        pos_attr = "pos_relaxed" if relaxed and hasattr(pyg, "pos_relaxed") else "pos"
        positions = getattr(pyg, pos_attr, None)
        if positions is None:
            raise ValueError(f"Pyg object lacks required '{pos_attr}' attribute.")

        numbers = self._ensure_numpy(pyg.atomic_numbers)
        pos_np = self._ensure_numpy(positions)
        cell_np = self._ensure_numpy(pyg.cell).squeeze(0)
        tags_np = self._ensure_numpy(pyg.tags)

        atoms = Atoms(
            numbers=numbers,
            positions=pos_np,
            cell=cell.Cell(cell_np),
            pbc=True,
            tags=tags_np,
        )

        fixed_tensor = getattr(pyg, "fixed", None)
        if fixed_tensor is not None:
            fixed_tensor = torch.as_tensor(fixed_tensor)
            mask = torch.nonzero(fixed_tensor == 1, as_tuple=False).squeeze()
            if mask.numel() == 0:
                fixed_atom_indices = []
            else:
                fixed_atom_indices = mask.tolist()
                if isinstance(fixed_atom_indices, int):
                    fixed_atom_indices = [fixed_atom_indices]
            if fixed_atom_indices:
                atoms.set_constraint(constraints.FixAtoms(indices=fixed_atom_indices))

        return atoms

    @staticmethod
    def _format_sample_id(sid: str) -> str:
        return sid

    def get_mapping_info(self, sid: str) -> Optional[Dict[str, Any]]:
        return self.mapping.get(sid)

    def get_interactions(self, atoms: Atoms):
        data = SiteAnalyzer(adslab=atoms)
        if data.center_atom is None:
            return None, None, None
        adsorbate_atom_positions = data.adsorbate_atom_positions
        connectivity = data._get_connectivity(data.atoms, data.cutoff_multiplier)
        interactions = []
        for bond in data.binding_info:
            bundle = [bond["adsorbate_element"]] + bond["slab_atom_elements"]
            typ = ["atop", "bridge", "hollow"][min(len(bond["slab_atom_elements"]), 3) - 1]
            bundle.append(typ)
            for idx in bond["slab_atom_idxs"]:
                second_int = data.second_binding_info[idx]
                bundle.append([second_int["slab_element"]] + second_int["second_interaction_element"])
            interactions.append(bundle)
        return interactions, adsorbate_atom_positions, connectivity

    @staticmethod
    def build_text(
        ads_sym: str,
        bulk_sym: str,
        miller_idx,
        interactions,
        adsorbate_positions,
    ) -> str:
        base = sanitize_ads_symbol(ads_sym) + "</s>" + bulk_sym + " " + str(tuple(miller_idx)) + "</s>"
        segments = []

        for bundle in interactions:
            if not bundle:
                return None

            first_list_idx = None
            for idx, item in enumerate(bundle):
                if isinstance(item, list):
                    first_list_idx = idx
                    break

            if first_list_idx is None or first_list_idx < 2:
                return None

            site_type = bundle[first_list_idx - 1]
            primary_atoms = bundle[1:first_list_idx - 1]
            secondary_lists = bundle[first_list_idx:]

            if not primary_atoms or not secondary_lists:
                return None

            primary_tokens = [bundle[0]] + list(primary_atoms) + [site_type]
            segment = "[" + " ".join(str(tok) for tok in primary_tokens) + "]"

            for sec in secondary_lists:
                segment += "[" + " ".join(str(tok) for tok in sec) + "]"

            segments.append(segment)

        if not segments:
            return None

        string = base + "".join(segments)
        rounded_array = np.round(np.array(adsorbate_positions), 3)
        for pos in rounded_array.tolist():
            string += str(pos)

        return string.replace(",", "").replace("*", "")

    def run(self):
        buckets = {split: self._init_bucket() for split in self.target_splits}
        sigma_threshold = self.energy_std * 5 if self.energy_std > 0 else None
        raw_split_counts = Counter()

        num_samples = len(self.lmdb_data)
        skip_stats: Counter = Counter()
        if self.limit is not None:
            limit = min(self.limit, num_samples)
            rng = np.random.default_rng(self.seed)
            sample_indices: Sequence[int] = rng.permutation(num_samples)[:limit]
        else:
            sample_indices = range(num_samples)

        skipped = 0
        processed_preview = 0
        processed_preview_limit = self.debug_print_limit if self.debug else 0
        pbar = tqdm.tqdm(sample_indices, total=len(sample_indices))
        for lmdb_idx in pbar:
            idx_int = int(lmdb_idx)
            try:
                pyg = self.lmdb_data[idx_int]
                if pyg is None:
                    skipped += 1
                    self._record_skip(skip_stats, "lmdb_none", idx=idx_int)
                    continue
            except Exception as exc:
                skipped += 1
                self._record_skip(skip_stats, "lmdb_exception", idx=idx_int, detail=str(exc))
                continue

            sid_raw = getattr(pyg, "sid", None)
            if isinstance(sid_raw, torch.Tensor):
                sid_raw = sid_raw.item()
            if sid_raw is None:
                skipped += 1
                self._record_skip(skip_stats, "sid_missing", idx=idx_int)
                continue
            sid_str = sid_raw if isinstance(sid_raw, str) else str(int(sid_raw))

            meta = self.get_mapping_info(sid_str)
            sid_used_for_meta = sid_str
            if meta is None and not sid_str.startswith("random"):
                sid_with_prefix = f"random{sid_str}"
                meta = self.get_mapping_info(sid_with_prefix)
                if meta is not None:
                    sid_used_for_meta = sid_with_prefix
            if meta is None:
                skipped += 1
                self._record_skip(skip_stats, "mapping_missing", idx=idx_int, sid=sid_str)
                continue
            sid_str = sid_used_for_meta

            split_value_raw = meta.get("split")
            split_value = str(split_value_raw).strip().lower() if split_value_raw is not None else "unknown"
            raw_split_counts[split_value] += 1
            if split_value not in buckets:
                continue
            bucket = buckets[split_value]

            ads_sym = meta.get("ads_symbols")
            bulk_sym = meta.get("bulk_symbols")
            miller_idx = meta.get("miller_index")
            if ads_sym is None or bulk_sym is None or miller_idx is None:
                skipped += 1
                self._record_skip(skip_stats, "meta_incomplete", idx=idx_int, sid=sid_str)
                continue

            try:
                atoms = self._atoms_from_pyg(pyg, relaxed=self.use_relaxed_positions)
            except ValueError as exc:
                skipped += 1
                self._record_skip(skip_stats, "atoms_relaxed_missing", idx=idx_int, sid=sid_str, detail=str(exc))
                continue
            try:
                init_atoms = self._atoms_from_pyg(pyg, relaxed=False)
            except ValueError as exc:
                skipped += 1
                self._record_skip(skip_stats, "atoms_init_missing", idx=idx_int, sid=sid_str, detail=str(exc))
                continue
            cell = np.asarray(init_atoms.get_cell().array, dtype=np.float32)
            pbc = np.asarray(init_atoms.get_pbc(), dtype=bool)

            tags_array = self._ensure_numpy(pyg.tags).astype(int).tolist()
            adsorbate_atom_idxs = [idx for idx, tag in enumerate(tags_array) if tag == 2]
            if len(adsorbate_atom_idxs) == 0:
                skipped += 1
                self._record_skip(skip_stats, "no_adsorbate_tag", idx=idx_int, sid=sid_str)
                continue

            element_all = atoms.get_chemical_symbols()
            natoms_all = len(element_all)
            if natoms_all >= self.max_text_len:
                skipped += 1
                self._record_skip(
                    skip_stats,
                    "atom_count_exceeds_text_len",
                    idx=idx_int,
                    sid=sid_str,
                    detail=str(natoms_all),
                )
                continue

            interactions, adsorbate_atom_positions, connectivity = self.get_interactions(init_atoms)
            if interactions is None:
                skipped += 1
                self._record_skip(skip_stats, "no_interactions", idx=idx_int, sid=sid_str)
                continue

            string = self.build_text(ads_sym, bulk_sym, miller_idx, interactions, adsorbate_atom_positions)
            if string is None:
                skipped += 1
                self._record_skip(skip_stats, "string4_missing", idx=idx_int, sid=sid_str)
                continue
            if len(string) >= self.max_text_len:
                skipped += 1
                self._record_skip(
                    skip_stats,
                    "text_exceeds_max_len",
                    idx=idx_int,
                    sid=sid_str,
                    detail=str(len(string)),
                )
                continue

            element = list(element_all)

            try:
                total_energy = self._extract_energy(pyg)
            except ValueError as exc:
                skipped += 1
                self._record_skip(skip_stats, "energy_missing", idx=idx_int, sid=sid_str, detail=str(exc))
                continue

            if sigma_threshold is not None and abs(total_energy - self.energy_mean) > sigma_threshold:
                bucket["outlier_log"].append(
                    f"sid={sid_str} energy={total_energy:.4f} deviates >5σ (mean={self.energy_mean:.4f}, std={self.energy_std:.4f})"
                )
                continue

            connectivity_np = np.asarray(connectivity)
            pos_init = np.asarray(init_atoms.get_positions(), dtype=np.float32)
            m = torch.zeros(natoms_all, dtype=torch.bool)
            if adsorbate_atom_idxs:
                m[adsorbate_atom_idxs] = True
            pos_tensor = torch.tensor(pos_init)

            sample_id = self._format_sample_id(sid_str)
            sid_key = str(sid_str)
            existing_idx = bucket["seen_sids"].get(sid_key)
            if existing_idx is not None:
                prev_energy = bucket["labels"][existing_idx]
                prev_delta = self._energy_delta(sid_key, prev_energy)
                new_delta = self._energy_delta(sid_key, total_energy)
                if new_delta < prev_delta:
                    bucket["strings"][existing_idx] = string
                    bucket["elements"][existing_idx] = element
                    bucket["connectivities"][existing_idx] = connectivity_np
                    bucket["positions"][existing_idx] = pos_tensor
                    bucket["pos_mask"][existing_idx] = m
                    bucket["cells"][existing_idx] = cell
                    bucket["pbcs"][existing_idx] = pbc
                    bucket["labels"][existing_idx] = total_energy
                    bucket["sids"][existing_idx] = sample_id
                    bucket["ads_labels"][existing_idx] = ads_sym
                    bucket["bulk_labels"][existing_idx] = bulk_sym
                    bucket["miller_labels"][existing_idx] = tuple(miller_idx)
                    bucket["duplicate_log"].append(
                        f"sid={sid_key} replaced previous entry (dE_old={prev_delta:.4f}, dE_new={new_delta:.4f})"
                    )
                else:
                    bucket["duplicate_log"].append(
                        f"sid={sid_key} skipped duplicate (dE_old={prev_delta:.4f} <= dE_new={new_delta:.4f})"
                    )
                continue

            bucket["seen_sids"][sid_key] = len(bucket["strings"])
            bucket["strings"].append(string)
            bucket["elements"].append(element)
            bucket["connectivities"].append(connectivity_np)
            bucket["positions"].append(pos_tensor)
            bucket["pos_mask"].append(m)
            bucket["cells"].append(cell)
            bucket["pbcs"].append(pbc)
            bucket["labels"].append(total_energy)
            bucket["sids"].append(sample_id)
            bucket["ads_labels"].append(ads_sym)
            bucket["bulk_labels"].append(bulk_sym)
            bucket["miller_labels"].append(tuple(miller_idx))

            if self.debug and processed_preview < processed_preview_limit:
                print(f"[DEBUG] kept idx={idx_int} sid={sid_str} natoms={natoms_all}")
                processed_preview += 1

        if skipped > 0:
            print(f"[WARN] skipped {skipped} items due to missing metadata or fields.")
        if self.debug and skip_stats:
            print("[DEBUG] Skip breakdown:")
            for reason, count in skip_stats.most_common():
                print(f"  - {reason}: {count}")

        final_split_counts = Counter()
        for split_name in self.target_splits:
            bucket = buckets[split_name]
            total_entries = len(bucket["strings"])
            if (
                self.split_sample_limit is not None
                and self.split_sample_limit > 0
                and total_entries > self.split_sample_limit
            ):
                rng = np.random.default_rng(self.seed)
                chosen_idx = np.sort(
                    rng.choice(total_entries, size=self.split_sample_limit, replace=False)
                )

                def select(lst):
                    return [lst[i] for i in chosen_idx]

                bucket["strings"] = select(bucket["strings"])
                bucket["elements"] = select(bucket["elements"])
                bucket["connectivities"] = select(bucket["connectivities"])
                bucket["positions"] = select(bucket["positions"])
                bucket["pos_mask"] = select(bucket["pos_mask"])
                bucket["cells"] = select(bucket["cells"])
                bucket["pbcs"] = select(bucket["pbcs"])
                bucket["labels"] = select(bucket["labels"])
                bucket["sids"] = select(bucket["sids"])
                bucket["ads_labels"] = select(bucket["ads_labels"])
                bucket["bulk_labels"] = select(bucket["bulk_labels"])
                bucket["miller_labels"] = select(bucket["miller_labels"])
                print(
                    f"[INFO] Down-sampled split '{split_name}' from {total_entries} to {len(bucket['strings'])} entries for split sampling."
                )

            kept_len = len(bucket["strings"])
            final_split_counts[split_name] = kept_len
            if not kept_len:
                continue

            if bucket["duplicate_log"]:
                self._write_log_file(f"duplicate_log_{split_name}.txt", bucket["duplicate_log"])
            if bucket["outlier_log"]:
                self._write_log_file(f"outlier_log_{split_name}.txt", bucket["outlier_log"])
            self._write_frequency_csv(
                bucket["ads_labels"],
                bucket["bulk_labels"],
                bucket["miller_labels"],
                suffix=split_name,
            )

            save_path = self.split_save_paths.get(split_name)
            if not save_path:
                save_path = os.path.join(self.output_dir, f"{split_name}_pa.pt")
            torch.save(
                (
                    bucket["strings"],
                    bucket["elements"],
                    bucket["connectivities"],
                    bucket["positions"],
                    bucket["pos_mask"],
                    bucket["labels"],
                    bucket["sids"],
                    bucket["cells"],
                    bucket["pbcs"],
                ),
                save_path,
            )
            print(f"[{split_name}] Processed {kept_len} samples. Saved to {save_path}")
            la = np.array(bucket["labels"], dtype=np.float32)
            print(f"[{split_name}] Energy stats mean={la.mean():.4f} eV | std={la.std():.4f}")

        self._write_split_summary(raw_split_counts, final_split_counts)

    def _record_skip(
        self,
        skip_stats: Counter,
        reason: str,
        idx: Optional[int] = None,
        sid: Optional[str] = None,
        detail: Optional[str] = None,
    ):
        skip_stats[reason] += 1
        if not self.debug:
            return
        count = skip_stats[reason]
        if count > self.debug_print_limit:
            return
        parts = [f"[DEBUG] skip[{reason}]"]
        if idx is not None:
            parts.append(f"idx={idx}")
        if sid is not None:
            parts.append(f"sid={sid}")
        if detail:
            parts.append(f"detail={detail}")
        print(" ".join(parts))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_path", type=str, required=True, help="OC20 LMDB path.")
    parser.add_argument("--mapping_path", type=str, required=True, help="oc20_data_mapping.pkl path.")
    parser.add_argument("--save_path", type=str, required=True, help="Output .pt file path.")
    parser.add_argument("--max_text_len", type=int, default=512, help="Maximum text length (characters).")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on processed samples.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed when --limit is used.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debugging output for skipped samples.",
    )
    parser.add_argument(
        "--debug_print_limit",
        type=int,
        default=5,
        help="Maximum number of debug lines to print per skip reason.",
    )
    parser.add_argument(
        "--split_names",
        type=str,
        default="train",
        help="Comma-separated list of OC20 splits to export (e.g., train,val_id,val_ood_ads,val_ood_cat,val_ood_both).",
    )
    parser.add_argument(
        "--split_save_dir",
        type=str,
        default=None,
        help="Directory to store per-split outputs when multiple splits are requested.",
    )
    parser.add_argument(
        "--split_sample_limit",
        type=int,
        default=None,
        help="Optional random sample limit applied after cleaning (useful for equal-sized ID/OOD subsets).",
    )
    parser.add_argument(
        "--split_summary_path",
        type=str,
        default=None,
        help="Optional CSV path to save raw vs kept counts for each split (defaults to save directory).",
    )
    args = parser.parse_args()

    split_names = None
    if getattr(args, "split_names", None):
        split_names = [item.strip() for item in args.split_names.split(",") if item.strip()]

    generator = OC20DataGenerator(
        lmdb_path=args.lmdb_path,
        mapping_path=args.mapping_path,
        save_path=args.save_path,
        max_text_len=args.max_text_len,
        limit=args.limit,
        seed=args.seed,
        debug=args.debug,
        debug_print_limit=args.debug_print_limit,
        split_names=split_names,
        split_save_dir=args.split_save_dir,
        split_sample_limit=args.split_sample_limit,
        split_summary_path=args.split_summary_path,
    )
    generator.run()


if __name__ == "__main__":
    main()
