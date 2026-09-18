"""
Postprocessing to prepare submission dataframe.

Postprocessor.select_candidates(...) produces CSR indices in ontology coordinate space (term positions),
i.e. each index is an integer in [0, n_terms) compatible with e.g. aspect_gt['ontology_term_ids'].

CSRState objects produced after Postprocessor.propagate(...) satisfy:
- for each row r, indices[indptr[r]:indptr[r+1]] are strictly increasing (sorted ascending with no duplicates).

Postprocessor.map_terms_to_pos(aspect_gt, term_ids) defines a one-to-one mapping from predicted term IDs
to ontology positions (no duplicated positions).
"""

import numpy as np
import pandas as pd

from dataclasses import dataclass

from .propagation import (
    align_csr_by_row_map,
    expand_with_children_csr,
    select_candidates_csr,
    drop_terms_csr,
    filter_terms_csr,
    propagate_max_up_csr,
    topk_from_csr,
    select_topk_state_csr
)
from .ensembling import (
    mean_inplace,
    merge_max_csr,
    merge_sum_count_csr,
)
from .debug import print_state_stats


@dataclass(frozen=True)
class CSRState:
    indptr: np.ndarray
    indices: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class PostprocessConfig:
    top_k: int = 500
    top_k_q: float = 0
    min_score: float = 0.01
    drop_zero_ia: bool = False
    exclude_not_descendants: bool = False
    drop_weak_preds: bool = False
    weak_min_train_n: int = 0
    weak_score_thr: float = 0.0


class Postprocessor:
    def __init__(
        self,
        config: PostprocessConfig,
        index_df: pd.DataFrame,
        prepared_gt: dict[str, dict],
    ) -> None:
        """
        Build a seq_key -> EntryID index for fast sequence-to-protein expansion.

        Internal state:
        - self._index_seq_keys: np.ndarray[object], shape (n_unique_seq,), sorted ascending.
          (Unique seq_key values used as the search domain for np.searchsorted)
        - self._index_indptr: np.ndarray[int64], shape (n_unique_seq + 1,).
          (CSR pointers into self._index_entry_ids; for a seq at position p, the associated
          proteins are self._index_entry_ids[self._index_indptr[p]:self._index_indptr[p+1]])
        - self._index_entry_ids: np.ndarray[object], shape (n_rows_index_df,).
          (EntryID values sorted by seq_key, aligned with _index_indptr segments)
        - self._topk_by_aspect: dict[str, int] - top_k values per aspect
        - self._prepared_gt: dict[str, dict] - prepared ground truth for each aspect
        """
        self.config = config

        entry_ids = index_df["EntryID"].to_numpy(dtype=object, copy=False)
        seq_keys = index_df["seq_key"].to_numpy(dtype=object, copy=False)

        order = np.argsort(seq_keys, kind="mergesort").astype(np.int64, copy=False)
        sorted_seq_keys = seq_keys[order]
        sorted_entry_ids = entry_ids[order]

        diff = sorted_seq_keys[1:] != sorted_seq_keys[:-1]
        boundaries = np.flatnonzero(diff).astype(np.int64, copy=False) + 1

        unique_seq_keys = np.empty(boundaries.size + 1, dtype=object)
        unique_seq_keys[0] = sorted_seq_keys[0]
        unique_seq_keys[1:] = sorted_seq_keys[boundaries]

        group_indptr = np.empty(unique_seq_keys.size + 1, dtype=np.int64)
        group_indptr[0] = 0
        group_indptr[1:-1] = boundaries
        group_indptr[-1] = int(sorted_seq_keys.size)

        self._index_seq_keys = unique_seq_keys
        self._index_indptr = group_indptr
        self._index_entry_ids = sorted_entry_ids

        self._prepared_gt = prepared_gt

        # Compute topk_by_aspect
        if config.top_k_q > 0:
            # Compute from prepared_gt using top_k_q
            self._topk_by_aspect = self._get_topk_by_aspect(q=config.top_k_q)
        else:
            # Use same top_k for all aspects
            self._topk_by_aspect = {aspect: int(config.top_k) for aspect in prepared_gt.keys()}

        print(f"[INFO] aspect top-k: {self._topk_by_aspect}")

    def _get_topk_by_aspect(
        self,
        q: float,
    ) -> dict[str, int]:
        """Get top-k for each aspect."""
        topk_by_aspect: dict[str, int] = {}
        for aspect_name in self._prepared_gt.keys():
            gt_pack = self._prepared_gt[aspect_name]["gt"]
            indptr = gt_pack["gt_indptr"]
            counts = (indptr[1:] - indptr[:-1])
            pos = int(np.ceil(q * float(counts.size - 1)))
            k = np.partition(counts, pos)[pos]
            topk_by_aspect[str(aspect_name)] = int(k)

        return topk_by_aspect

    def map_seqs_to_prots(
        self,
        seq_keys: np.ndarray,
        preds: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Map sequences to proteins, by duplicating predictions for respective sequences.
        """
        pos = np.searchsorted(self._index_seq_keys, seq_keys, side="left")
        in_bounds = pos < int(self._index_seq_keys.size)
        matched = np.zeros(int(seq_keys.size), dtype=bool)

        if bool(np.any(in_bounds)):
            idx = pos[in_bounds].astype(np.int64, copy=False)
            matched[in_bounds] = self._index_seq_keys[idx] == seq_keys[in_bounds]

        total = 0
        for i in range(int(seq_keys.size)):
            if not bool(matched[i]):
                continue
            p = int(pos[i])
            total += int(self._index_indptr[p + 1] - self._index_indptr[p])

        if total == 0:
            return np.empty(0, dtype=object), preds[:0]

        n_pred = int(preds.shape[1])
        entry_ids = np.empty(total, dtype=object)
        out_preds = np.empty((total, n_pred), dtype=preds.dtype)

        out_pos = 0
        for i in range(int(seq_keys.size)):
            if not bool(matched[i]):
                continue
            p = int(pos[i])
            start = int(self._index_indptr[p])
            end = int(self._index_indptr[p + 1])
            count = end - start
            entry_ids[out_pos:out_pos + count] = self._index_entry_ids[start:end]
            out_preds[out_pos:out_pos + count, :] = preds[i, :]
            out_pos += count

        return entry_ids, out_preds

    def map_terms_to_pos(self, aspect_name: str, term_ids: np.ndarray) -> np.ndarray:
        """Map terms to positions in ontology (np.int32)."""
        aspect_gt = self._prepared_gt[aspect_name]
        ontology_term_ids = aspect_gt["ontology_term_ids"]
        term_to_pos = {t: i for i, t in enumerate(ontology_term_ids.tolist())}

        term_pos = np.empty(term_ids.size, dtype=np.int32)
        for i in range(int(term_ids.size)):
            t = term_ids[i]
            if t not in term_to_pos:
                raise ValueError(f"predicted term not in ontology: {t}")
            term_pos[i] = np.int32(term_to_pos[t])

        return term_pos

    @staticmethod
    def _map_entry_ids_to_rows(row_ids: np.ndarray, entry_ids: np.ndarray) -> np.ndarray:
        """Map each EntryID to its row index in a sorted row_ids array (np.int64); -1 if absent."""
        if int(entry_ids.size) == 0 or int(row_ids.size) == 0:
            return -np.ones(int(entry_ids.size), dtype=np.int64)

        pos = np.searchsorted(row_ids, entry_ids, side="left")
        in_bounds = pos < int(row_ids.size)

        matched = np.zeros(int(entry_ids.size), dtype=bool)
        matched[in_bounds] = row_ids[pos[in_bounds]] == entry_ids[in_bounds]

        row_map = -np.ones(int(entry_ids.size), dtype=np.int64)
        row_map[matched] = pos[matched]
        return row_map

    def prepare_extra_terms(
        self,
        aspect_name: str,
        entry_ids: np.ndarray,
        source: str,
        score_value: float = 1.0,
    ) -> CSRState:
        """Prepare external terms as CSRState with constant score."""
        if source not in ("test_nonexp", "test_exp"):
            raise ValueError(f"Unexpected source={source!r}")

        aspect_gt = self._prepared_gt[aspect_name]
        src = aspect_gt[source]
        row_ids = src[f"{source}_ids"]
        row_map = self._map_entry_ids_to_rows(row_ids, entry_ids)

        indptr, indices = align_csr_by_row_map(
            row_map=row_map,
            indptr=src[f"{source}_indptr"],
            indices=src[f"{source}_indices"],
        )
        scores = np.full(int(indices.size), np.float32(score_value), dtype=np.float32)
        return CSRState(indptr=indptr, indices=indices, scores=scores)

    def select_candidates(
        self,
        term_pos: np.ndarray,
        preds: np.ndarray,
    ) -> CSRState:
        """Select candidates by min_score. Assumes np.int/float32 for inputs."""
        indptr, indices, scores = select_candidates_csr(
            term_pos=term_pos,
            preds=preds,
            min_score=float(self.config.min_score),
        )
        return CSRState(indptr=indptr, indices=indices, scores=scores)

    def drop_not_terms(
        self,
        state: CSRState,
        aspect_name: str,
        entry_ids: np.ndarray,
    ) -> CSRState:
        """Drop not-terms from state."""
        aspect_gt = self._prepared_gt[aspect_name]
        graph = aspect_gt["graph"]
        not_terms = aspect_gt["not_terms"]

        row_map = self._map_entry_ids_to_rows(not_terms["not_ids"], entry_ids)

        drop_indptr, drop_indices = align_csr_by_row_map(
            row_map=row_map,
            indptr=not_terms["not_indptr"],
            indices=not_terms["not_indices"],
        )

        if bool(self.config.exclude_not_descendants):
            drop_indptr, drop_indices = expand_with_children_csr(
                indptr=drop_indptr,
                indices=drop_indices,
                children_indptr=graph["children_indptr"],
                children_indices=graph["children_indices"],
            )

        indptr, indices, scores = drop_terms_csr(
            indptr=state.indptr,
            indices=state.indices,
            scores=state.scores,
            drop_indptr=drop_indptr,
            drop_indices=drop_indices,
        )
        return CSRState(indptr=indptr, indices=indices, scores=scores)

    def drop_known_terms(
        self,
        state: CSRState,
        aspect_name: str,
        entry_ids: np.ndarray
    ) -> CSRState:
        aspect_gt = self._prepared_gt[aspect_name]
        known = aspect_gt["known_terms"]

        row_map = self._map_entry_ids_to_rows(known["known_ids"], entry_ids)

        drop_indptr, drop_indices = align_csr_by_row_map(
            row_map=row_map,
            indptr=known["known_indptr"],
            indices=known["known_indices"],
        )

        indptr, indices, scores = drop_terms_csr(
            indptr=state.indptr,
            indices=state.indices,
            scores=state.scores,
            drop_indptr=drop_indptr,
            drop_indices=drop_indices,
        )
        return CSRState(indptr=indptr, indices=indices, scores=scores)

    def apply_filters(
        self,
        state: CSRState,
        aspect_name: str
    ) -> CSRState:
        """Apply filters (IA, train counts + pred threshold) to state."""
        aspect_gt = self._prepared_gt[aspect_name]
        n_terms = aspect_gt["ontology_term_ids"].size

        if bool(self.config.drop_zero_ia):
            ia_eff = aspect_gt["ia"]
        else:
            ia_eff = np.ones(n_terms, dtype=np.float32)

        # calculate frequency of terms in train set
        train_counts = np.bincount(aspect_gt["gt"]["gt_indices"], minlength=n_terms)

        if bool(self.config.drop_weak_preds):
            min_n = int(self.config.weak_min_train_n)
            thr = float(self.config.weak_score_thr)
        else:
            min_n = 0
            thr = -1.0

        indptr, indices, scores = filter_terms_csr(
            indptr=state.indptr,
            indices=state.indices,
            scores=state.scores,
            ia=ia_eff,
            train_counts=train_counts,
            weak_min_train_n=min_n,
            weak_score_thr=thr,
        )
        return CSRState(indptr=indptr, indices=indices, scores=scores)

    def propagate(
        self,
        state: CSRState,
        aspect_name: str,
    ) -> CSRState:
        """Propagate max to ancestors."""
        aspect_gt = self._prepared_gt[aspect_name]
        graph = aspect_gt["graph"]
        indptr, indices, scores = propagate_max_up_csr(
            indptr=state.indptr,
            indices=state.indices,
            scores=state.scores,
            parents_indptr=graph["parents_indptr"],
            parents_indices=graph["parents_indices"],
        )
        return CSRState(indptr=indptr, indices=indices, scores=scores)

    def topk(
        self,
        state: CSRState,
        aspect_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute top-k arrays."""
        topk_pos, topk_scores = topk_from_csr(
            indptr=state.indptr,
            indices=state.indices,
            scores=state.scores,
            top_k=self._topk_by_aspect[aspect_name],
        )
        return topk_pos, topk_scores

    def topk_state(self, state: CSRState, aspect_name: str) -> CSRState:
        indptr, indices, scores = select_topk_state_csr(
            indptr=state.indptr,
            indices=state.indices,
            scores=state.scores,
            top_k=int(self._topk_by_aspect[aspect_name]),
            max_k=int(self.config.top_k),
        )
        return CSRState(indptr=indptr, indices=indices, scores=scores)

    def postprocess_state(
        self,
        state: CSRState,
        data_type: str,
        entry_ids: np.ndarray,
        aspect_name: str,
        propagate: bool = True,
        add_nonexp_terms: bool = False,
        add_exp_terms: bool = False,
        drop_known: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Postprocess state and return topk_pos, topk_scores.
        """
        state = CSRState(
            indptr=state.indptr.copy(),
            indices=state.indices.copy(),
            scores=state.scores.copy(),
        )

        state = self.drop_not_terms(state, aspect_name, entry_ids)
        print_state_stats(state, "After drop_not_terms")

        if propagate:
            state = self.propagate(state, aspect_name)
            print_state_stats(state, "After propagate")

        if data_type == "test":
            # add extra terms (with score=1.0 by default)
            if add_nonexp_terms:
                extra = self.prepare_extra_terms(aspect_name, entry_ids, "test_nonexp")
                state = self.blend_states([state, extra], mode="mean")
                print_state_stats(state, "After blend_states (nonexp)")

            if add_exp_terms:
                extra = self.prepare_extra_terms(aspect_name, entry_ids, "test_exp")
                state = self.blend_states([state, extra], mode="max")
                print_state_stats(state, "After blend_states (exp)")

            # drop not-terms, propagate, drop known terms, apply filters
            if add_nonexp_terms or add_exp_terms:
                state = self.drop_not_terms(state, aspect_name, entry_ids)
                print_state_stats(state, "After drop_not_terms (after blend)")
                state = self.propagate(state, aspect_name)
                print_state_stats(state, "After propagate (after blend)")

        if drop_known:
            state = self.drop_known_terms(state, aspect_name, entry_ids)
            print_state_stats(state, "After drop_known_terms")

        state = self.apply_filters(state, aspect_name)
        print_state_stats(state, "After apply_filters")

        # select top-k terms per protein
        topk_pos, topk_scores = self.topk(state, aspect_name=aspect_name)

        return topk_pos, topk_scores

    @staticmethod
    def blend_states(states: list[CSRState], mode: str) -> CSRState:
        """Blend states by max or mean."""
        if len(states) == 0:
            raise ValueError("states is empty")

        if mode == "max":
            acc = states[0]
            for st in states[1:]:
                indptr, indices, scores = merge_max_csr(
                    acc.indptr, acc.indices, acc.scores,
                    st.indptr, st.indices, st.scores,
                )
                acc = CSRState(indptr=indptr, indices=indices, scores=scores)
            return acc

        if mode != "mean":
            raise ValueError(f"Unexpected mode={mode!r}")

        base = states[0]
        sum_indptr = base.indptr
        sum_indices = base.indices
        sum_scores = base.scores
        sum_counts = np.ones(int(sum_indices.size), dtype=np.int16)

        for st in states[1:]:
            sum_indptr, sum_indices, sum_scores, sum_counts = merge_sum_count_csr(
                sum_indptr, sum_indices, sum_scores, sum_counts,
                st.indptr, st.indices, st.scores,
            )

        mean_inplace(sum_scores, sum_counts)
        return CSRState(indptr=sum_indptr, indices=sum_indices, scores=sum_scores)
