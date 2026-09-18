import numba as nb
import numpy as np

from src.core.postprocess import CSRState


def map_prots_to_member_pos(axis_entry_ids: np.ndarray, member_entry_ids: np.ndarray) -> np.ndarray:
    order = np.argsort(member_entry_ids, kind="mergesort").astype(np.int64, copy=False)
    sorted_ids = member_entry_ids[order]
    pos = np.searchsorted(sorted_ids, axis_entry_ids, side="left").astype(np.int64, copy=False)
    in_bounds = pos < int(sorted_ids.size)
    out = -np.ones(int(axis_entry_ids.size), dtype=np.int64)
    if bool(np.any(in_bounds)):
        cand = pos[in_bounds]
        ok = sorted_ids[cand] == axis_entry_ids[in_bounds]
        axis_pos = np.flatnonzero(in_bounds)[ok]
        out[axis_pos] = order[cand[ok]]
    return out


def align_state_to_protein_axis(axis_entry_ids: np.ndarray, member_entry_ids: np.ndarray, state: CSRState) -> CSRState:
    axis_to_member = map_prots_to_member_pos(axis_entry_ids, member_entry_ids)
    n_axis = int(axis_entry_ids.size)
    counts = np.zeros(n_axis, dtype=np.int64)
    for axis_pos in range(n_axis):
        member_pos = int(axis_to_member[axis_pos])
        if member_pos >= 0:
            counts[axis_pos] = state.indptr[member_pos + 1] - state.indptr[member_pos]
    out_indptr = np.empty(n_axis + 1, dtype=np.int64)
    out_indptr[0] = 0
    np.cumsum(counts, out=out_indptr[1:])
    out_indices = np.empty(int(out_indptr[-1]), dtype=state.indices.dtype)
    out_scores = np.empty(int(out_indptr[-1]), dtype=state.scores.dtype)
    for axis_pos in range(n_axis):
        member_pos = int(axis_to_member[axis_pos])
        if member_pos < 0:
            continue
        s = int(state.indptr[member_pos])
        e = int(state.indptr[member_pos + 1])
        if s == e:
            continue
        o = int(out_indptr[axis_pos])
        out_indices[o:o + e - s] = state.indices[s:e]
        out_scores[o:o + e - s] = state.scores[s:e]
    return CSRState(indptr=out_indptr, indices=out_indices, scores=out_scores)


@nb.njit
def _union_count(indptrs, indices_list, prot_idx: int) -> int:
    n_members = len(indptrs)
    curs = np.empty(n_members, dtype=np.int64)
    ends = np.empty(n_members, dtype=np.int64)
    for m in range(n_members):
        curs[m] = indptrs[m][prot_idx]
        ends[m] = indptrs[m][prot_idx + 1]
    out = 0
    while True:
        min_term = 2147483647
        any_left = False
        for m in range(n_members):
            if curs[m] < ends[m]:
                any_left = True
                term = int(indices_list[m][curs[m]])
                if term < min_term:
                    min_term = term
        if not any_left:
            break
        out += 1
        for m in range(n_members):
            while curs[m] < ends[m] and int(indices_list[m][curs[m]]) == min_term:
                curs[m] += 1
    return out


@nb.njit
def pack_union_dataset(prots_idx: np.ndarray, indptrs, indices_list, scores_list):
    n_in = int(prots_idx.size)
    n_members = len(indptrs)
    n_queries = 0
    nnz_total = 0
    for i in range(n_in):
        cnt = _union_count(indptrs, indices_list, int(prots_idx[i]))
        if cnt:
            n_queries += 1
            nnz_total += cnt
    groups = np.empty(n_queries, dtype=np.int32)
    prots_out = np.empty(n_queries, dtype=np.int32)
    term_indptr = np.empty(n_queries + 1, dtype=np.int64)
    term_indices = np.empty(nnz_total, dtype=np.int32)
    member_scores = np.empty((nnz_total, n_members), dtype=np.float32)
    term_indptr[0] = 0
    q = 0
    off = 0
    for i in range(n_in):
        prot_idx = int(prots_idx[i])
        cnt = _union_count(indptrs, indices_list, prot_idx)
        if not cnt:
            continue
        groups[q] = cnt
        prots_out[q] = prot_idx
        term_indptr[q + 1] = term_indptr[q] + cnt
        curs = np.empty(n_members, dtype=np.int64)
        ends = np.empty(n_members, dtype=np.int64)
        for m in range(n_members):
            curs[m] = indptrs[m][prot_idx]
            ends[m] = indptrs[m][prot_idx + 1]
        for _ in range(cnt):
            min_term = 2147483647
            for m in range(n_members):
                if curs[m] < ends[m]:
                    term = int(indices_list[m][curs[m]])
                    if term < min_term:
                        min_term = term
            term_indices[off] = min_term
            for m in range(n_members):
                value = np.float32(0.0)
                if curs[m] < ends[m] and int(indices_list[m][curs[m]]) == min_term:
                    value = np.float32(scores_list[m][curs[m]])
                    while curs[m] < ends[m] and int(indices_list[m][curs[m]]) == min_term:
                        curs[m] += 1
                member_scores[off, m] = value
            off += 1
        q += 1
    return groups, prots_out, term_indptr, term_indices, member_scores
