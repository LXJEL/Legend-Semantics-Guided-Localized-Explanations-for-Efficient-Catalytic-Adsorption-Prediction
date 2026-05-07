"""Site-analysis helpers for determining how an adsorbate is bound to a slab."""

import numpy as np
from ase import neighborlist, Atoms
from ase.neighborlist import natural_cutoffs
from scipy.spatial.distance import euclidean
from itertools import combinations


class SiteAnalyzer:
    def __init__(self, adslab, cutoff_multiplier=1.0):
        """
        Initialize the site-analysis helper.

        Args:
            adslab (ase.Atoms): slab-plus-adsorbate structure to analyze.
        """
        self.atoms = adslab
        self.cutoff_multiplier = cutoff_multiplier
        self.binding_info = self._find_binding_graph()
        self.second_binding_info = self._find_second_binding_graph()
        self.center_atom, self.adsorbate_atom_positions, self.adsorbate_atom_type = self._get_center_ads_atom()
        if self.center_atom is not None:
            self.center_binding_info = self._find_binding_atoms_from_center()

    def _get_center_ads_atom(self):
        """
        Identify the central atom of the adsorbate.
        """
        tags = self.atoms.get_tags()
        elements = self.atoms.get_chemical_symbols()

        # Tag 2 marks adsorbate atoms in this dataset.
        adsorbate_atom_idxs = [idx for idx, tag in enumerate(tags) if tag == 2]
        if not adsorbate_atom_idxs:
            return None, None, None

        adsorbate_atom_type = [elements[idx] for idx in adsorbate_atom_idxs]
        adsorbate_atom_positions = self.atoms.get_positions()[adsorbate_atom_idxs]
        center_position = np.mean(adsorbate_atom_positions, axis=0)
        all_distance = [euclidean(center_position, position) for position in adsorbate_atom_positions]

        if not all_distance:
            return None, None, None

        min_idx = adsorbate_atom_idxs[np.argmin(all_distance)]
        center_atom = self.atoms[min_idx]
        return center_atom, adsorbate_atom_positions, adsorbate_atom_type

    def _find_binding_atoms_from_center(self):
        """
        Find the surface atoms that bind to the adsorbate center atom.
        """
        tags = self.atoms.get_tags()
        elements = self.atoms.get_chemical_symbols()

        # Tag 1 marks slab/surface atoms.
        slab_atom_idxs = [idx for idx, tag in enumerate(tags) if tag == 1]
        connectivity = self._get_connectivity(self.atoms, self.cutoff_multiplier)
        binding_info = []
        adslab_positions = self.atoms.get_positions()

        if sum(connectivity[self.center_atom.index][slab_atom_idxs]) >= 1:
            bound_slab_idxs = [
                idx_slab
                for idx_slab in slab_atom_idxs
                if connectivity[self.center_atom.index][idx_slab] == 1
            ]
            ads_idx_info = {
                "adsorbate_idx": self.center_atom.index,
                "adsorbate_element": elements[self.center_atom.index],
                "slab_atom_elements": [
                    element
                    for idx_el, element in enumerate(elements)
                    if idx_el in bound_slab_idxs
                ],
                "slab_atom_idxs": bound_slab_idxs,
                "bound_position": adslab_positions[self.center_atom.index],
            }
            binding_info.append(ads_idx_info)
        return binding_info

    def _find_second_binding_graph(self):
        elements = self.atoms.get_chemical_symbols()
        connectivity = self._get_connectivity(self.atoms, self.cutoff_multiplier)

        second_binding_info = {}
        for interaction in self.binding_info:
            slab_atom_idxs = interaction["slab_atom_idxs"]
            if len(slab_atom_idxs) != 0:
                for idx in slab_atom_idxs:
                    second_connection = connectivity[idx] == 1
                    second_int_idx = np.where(second_connection)[0]
                    second_int_info = {
                        "slab_element": elements[idx],
                        "second_interaction_idx": second_int_idx,
                        "second_interaction_element": [elements[idx] for idx in second_int_idx],
                    }
                    second_binding_info.update({idx: second_int_info})
        return second_binding_info

    def _find_binding_graph(self):
        tags = self.atoms.get_tags()
        elements = self.atoms.get_chemical_symbols()

        # Split adsorbate and slab indices based on the tag convention.
        adsorbate_atom_idxs = [idx for idx, tag in enumerate(tags) if tag == 2]
        slab_atom_idxs = [idx for idx, tag in enumerate(tags) if tag != 2]

        connectivity = self._get_connectivity(self.atoms, self.cutoff_multiplier)

        binding_info = []
        adslab_positions = self.atoms.get_positions()
        for idx in adsorbate_atom_idxs:
            if sum(connectivity[idx][slab_atom_idxs]) >= 1:
                bound_slab_idxs = [
                    idx_slab for idx_slab in slab_atom_idxs if connectivity[idx][idx_slab] == 1
                ]
                ads_idx_info = {
                    "adsorbate_idx": idx,
                    "adsorbate_element": elements[idx],
                    "slab_atom_elements": [element for idx_el, element in enumerate(elements) if idx_el in bound_slab_idxs],
                    "slab_atom_idxs": bound_slab_idxs,
                    "bound_position": adslab_positions[idx],
                }
                binding_info.append(ads_idx_info)
        return binding_info

    def _get_connectivity(self, atoms, cutoff_multiplier=1.0):
        """
        Build the atom connectivity matrix.

        Args:
            atoms (ase.Atoms): structure to analyze.
            cutoff_multiplier (float): scale factor for the natural cutoffs.
        """
        cutoff = natural_cutoffs(atoms, mult=cutoff_multiplier)
        neighbor_list = neighborlist.NeighborList(
            cutoff,
            self_interaction=False,
            bothways=True,
            skin=0.05,
        )
        neighbor_list.update(atoms)
        matrix = neighborlist.get_connectivity_matrix(neighbor_list.nl).toarray()
        return matrix

    def get_dentate(self):
        """
        Return the number of adsorbate atoms bound to the surface.
        """
        return len(self.binding_info)

    def get_site_types(self):
        """
        Return the number of surface atoms contacted by each bound adsorbate atom.
        """
        return [len(binding["slab_atom_idxs"]) for binding in self.binding_info]

    def get_bound_atom_positions(self):
        """
        Return the coordinates of all bound adsorbate atoms.
        """
        positions = []
        for atom in self.binding_info:
            positions.append(atom["bound_position"])
        return positions

    def get_minimum_site_proximity(self, site_to_compare):
        """
        Return the minimum distance between the bound atoms of two adsorbates.
        """
        this_positions = self.get_bound_atom_positions()
        other_positions = site_to_compare.get_bound_atom_positions()
        distances = []
        if len(this_positions) > 0 and len(other_positions) > 0:
            for this_position in this_positions:
                for other_position in other_positions:
                    fake_atoms = Atoms("CO", positions=[this_position, other_position])
                    distances.append(fake_atoms.get_distance(0, 1, mic=True))
            return min(distances)
        return np.nan

    def get_adsorbate_bond_lengths(self):
        """Return adsorbate-adsorbate and adsorbate-surface bond lengths."""
        bond_lengths = {"adsorbate-adsorbate": {}, "adsorbate-surface": {}}
        adsorbate = self.atoms[
            [idx for idx, tag in enumerate(self.atoms.get_tags()) if tag == 2]
        ]
        adsorbate_connectivity = self._get_connectivity(adsorbate)
        combos = list(combinations(range(len(adsorbate)), 2))
        for combo in combos:
            if adsorbate_connectivity[combo[0], combo[1]] == 1:
                bond_lengths["adsorbate-adsorbate"][tuple(np.sort(combo))] = adsorbate.get_distance(
                    combo[0], combo[1], mic=True
                )

        for ads_info in self.binding_info:
            adsorbate_idx = ads_info["adsorbate_idx"]
            bond_lengths["adsorbate-surface"][adsorbate_idx] = [
                self.atoms.get_distances(adsorbate_idx, slab_idx, mic=True)[0]
                for slab_idx in ads_info["slab_atom_idxs"]
            ]

        return bond_lengths
