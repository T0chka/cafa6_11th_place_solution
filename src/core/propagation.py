"""
Ontology score propagation and top-k selection.

Invariants guaranteed by source data:
- The ontology parent graph is a DAG (no cycles reachable via parents_* CSR).

Invariants guaranteed upstream when building the ground-truth pickle / artifacts:
- parents_indptr/children_indptr have length n_terms + 1 and form valid CSR pointers:
  indptr[0]=0, indptr is non-decreasing, indptr[-1]=len(indices)
- parents_indices/children_indices store valid term indices in [0, n_terms)
- For each CSR slot, row-wise indices are sorted ascending (enables binary search)
- *_ids arrays are sorted unique; alignment to evaluated protein order uses *_row maps,
  where -1 denotes "no corresponding row".

Invariants guaranteed by postprocessing:
- Model outputs are aligned with the ontology vocabulary:
  term_pos has length n_pred and maps each prediction column to a term index in
  [0, n_terms) (map_terms_to_pos).
"""

import numpy as np
import numba as nb

from numba.typed import Dict, List
from numba import int32, float32


@nb.njit(cache=True)
def align_csr_by_row_map(
    row_map: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Align CSR by row map."""
    n_rows = int(row_map.size)
    counts = np.zeros(n_rows, dtype=np.int64)

    for row_index in range(n_rows):
        r = int(row_map[row_index])
        if r < 0:
            continue
        counts[row_index] = int(indptr[r + 1] - indptr[r])

    out_indptr = np.empty(n_rows + 1, dtype=np.int64)
    out_indptr[0] = 0
    for i in range(n_rows):
        out_indptr[i + 1] = out_indptr[i] + counts[i]

    total = int(out_indptr[n_rows])
    out_indices = np.empty(total, dtype=np.int32)

    for row_index in range(n_rows):
        r = int(row_map[row_index])
        if r < 0:
            continue
        start = int(indptr[r])
        end = int(indptr[r + 1])
        out_pos = int(out_indptr[row_index])
        m = end - start
        if m > 0:
            out_indices[out_pos:out_pos + m] = indices[start:end]

    return out_indptr, out_indices


@nb.njit(cache=True)
def expand_with_children_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    children_indptr: np.ndarray,
    children_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = int(indptr.size - 1)
    counts = np.zeros(n_rows, dtype=np.int64)

    for row_index in range(n_rows):
        seen = Dict.empty(key_type=int32, value_type=int32)
        queue = List.empty_list(int32)

        start = int(indptr[row_index])
        end = int(indptr[row_index + 1])
        for pos in range(start, end):
            term = int32(indices[pos])
            if seen.get(term, int32(0)) == int32(0):
                seen[term] = int32(1)
                queue.append(term)

        qh = 0
        while qh < len(queue):
            term = queue[qh]
            qh += 1
            cs = int(children_indptr[int(term)])
            ce = int(children_indptr[int(term) + 1])
            for p in range(cs, ce):
                child = int32(children_indices[p])
                if seen.get(child, int32(0)) == int32(0):
                    seen[child] = int32(1)
                    queue.append(child)

        counts[row_index] = len(seen)

    out_indptr = np.empty(n_rows + 1, dtype=np.int64)
    out_indptr[0] = 0
    for i in range(n_rows):
        out_indptr[i + 1] = out_indptr[i] + counts[i]

    total = int(out_indptr[n_rows])
    out_indices = np.empty(total, dtype=np.int32)

    for row_index in range(n_rows):
        seen = Dict.empty(key_type=int32, value_type=int32)
        queue = List.empty_list(int32)

        start = int(indptr[row_index])
        end = int(indptr[row_index + 1])
        for pos in range(start, end):
            term = int32(indices[pos])
            if seen.get(term, int32(0)) == int32(0):
                seen[term] = int32(1)
                queue.append(term)

        qh = 0
        while qh < len(queue):
            term = queue[qh]
            qh += 1
            cs = int(children_indptr[int(term)])
            ce = int(children_indptr[int(term) + 1])
            for p in range(cs, ce):
                child = int32(children_indices[p])
                if seen.get(child, int32(0)) == int32(0):
                    seen[child] = int32(1)
                    queue.append(child)

        m = len(seen)
        tmp = np.empty(m, dtype=np.int32)
        t = 0
        for term, flag in seen.items():
            if int(flag) == 1:
                tmp[t] = int32(term)
                t += 1

        tmp = np.sort(tmp[:t])
        out_pos = int(out_indptr[row_index])
        out_indices[out_pos:out_pos + t] = tmp

    return out_indptr, out_indices


@nb.njit(cache=True)
def select_candidates_csr(
    term_pos: np.ndarray,
    preds: np.ndarray,
    min_score: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select candidates by min_score as CSR."""
    n_rows = int(preds.shape[0])
    n_pred = int(preds.shape[1])
    min_score_f = float32(min_score)

    counts = np.zeros(n_rows, dtype=np.int64)
    for row_index in range(n_rows):
        c = 0
        for pred_index in range(n_pred):
            if float32(preds[row_index, pred_index]) > min_score_f:
                c += 1
        counts[row_index] = c

    indptr = np.empty(n_rows + 1, dtype=np.int64)
    indptr[0] = 0
    for i in range(n_rows):
        indptr[i + 1] = indptr[i] + counts[i]

    total = int(indptr[n_rows])
    indices = np.empty(total, dtype=np.int32)
    scores = np.empty(total, dtype=np.float32)

    tmp_terms = np.empty(n_pred, dtype=np.int32)
    tmp_scores = np.empty(n_pred, dtype=np.float32)

    for row_index in range(n_rows):
        m = 0
        for pred_index in range(n_pred):
            score = float32(preds[row_index, pred_index])
            if score <= min_score_f:
                continue
            tmp_terms[m] = int32(term_pos[pred_index])
            tmp_scores[m] = score
            m += 1

        if m == 0:
            continue

        order = np.argsort(tmp_terms[:m])
        out_pos = int(indptr[row_index])
        for j in range(m):
            k = int(order[j])
            indices[out_pos] = int32(tmp_terms[k])
            scores[out_pos] = float32(tmp_scores[k])
            out_pos += 1

    return indptr, indices, scores


@nb.njit(cache=True)
def drop_terms_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    scores: np.ndarray,
    drop_indptr: np.ndarray,
    drop_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop not-terms set from CSR."""
    n_rows = int(indptr.size - 1)
    out_counts = np.zeros(n_rows, dtype=np.int64)

    for row_index in range(n_rows):
        a0 = int(indptr[row_index])
        a1 = int(indptr[row_index + 1])
        b0 = int(drop_indptr[row_index])
        b1 = int(drop_indptr[row_index + 1])

        i = a0
        j = b0
        kept = 0

        while i < a1 and j < b1:
            ta = int(indices[i])
            tb = int(drop_indices[j])
            if ta == tb:
                i += 1
            elif ta < tb:
                kept += 1
                i += 1
            else:
                j += 1

        kept += (a1 - i)
        out_counts[row_index] = kept

    out_indptr = np.empty(n_rows + 1, dtype=np.int64)
    out_indptr[0] = 0
    for i in range(n_rows):
        out_indptr[i + 1] = out_indptr[i] + out_counts[i]

    total = int(out_indptr[n_rows])
    out_indices = np.empty(total, dtype=np.int32)
    out_scores = np.empty(total, dtype=np.float32)

    for row_index in range(n_rows):
        a0 = int(indptr[row_index])
        a1 = int(indptr[row_index + 1])
        b0 = int(drop_indptr[row_index])
        b1 = int(drop_indptr[row_index + 1])

        i = a0
        j = b0
        out_pos = int(out_indptr[row_index])

        while i < a1 and j < b1:
            ta = int(indices[i])
            tb = int(drop_indices[j])
            if ta == tb:
                i += 1
            elif ta < tb:
                out_indices[out_pos] = int32(ta)
                out_scores[out_pos] = float32(scores[i])
                out_pos += 1
                i += 1
            else:
                j += 1

        while i < a1:
            ta = int(indices[i])
            out_indices[out_pos] = int32(ta)
            out_scores[out_pos] = float32(scores[i])
            out_pos += 1
            i += 1

    return out_indptr, out_indices, out_scores


@nb.njit(cache=True)
def propagate_max_up_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    scores: np.ndarray,
    parents_indptr: np.ndarray,
    parents_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate max to ancestors as CSR."""
    n_rows = int(indptr.size - 1)
    out_counts = np.zeros(n_rows, dtype=np.int64)

    for row_index in range(n_rows):
        score_map = Dict.empty(key_type=int32, value_type=float32)
        queue = List.empty_list(int32)

        start = int(indptr[row_index])
        end = int(indptr[row_index + 1])
        for pos in range(start, end):
            term = int32(indices[pos])
            score = float32(scores[pos])
            old = score_map.get(term, float32(-np.inf))
            if score > old:
                score_map[term] = score
                queue.append(term)

        qh = 0
        while qh < len(queue):
            term = queue[qh]
            qh += 1
            score = score_map.get(term, float32(-np.inf))
            ps = int(parents_indptr[int(term)])
            pe = int(parents_indptr[int(term) + 1])
            for p in range(ps, pe):
                parent = int32(parents_indices[p])
                old = score_map.get(parent, float32(-np.inf))
                if score > old:
                    score_map[parent] = score
                    queue.append(parent)

        out_counts[row_index] = len(score_map)

    out_indptr = np.empty(n_rows + 1, dtype=np.int64)
    out_indptr[0] = 0
    for i in range(n_rows):
        out_indptr[i + 1] = out_indptr[i] + out_counts[i]

    total = int(out_indptr[n_rows])
    out_indices = np.empty(total, dtype=np.int32)
    out_scores = np.empty(total, dtype=np.float32)

    for row_index in range(n_rows):
        score_map = Dict.empty(key_type=int32, value_type=float32)
        queue = List.empty_list(int32)

        start = int(indptr[row_index])
        end = int(indptr[row_index + 1])
        for pos in range(start, end):
            term = int32(indices[pos])
            score = float32(scores[pos])
            old = score_map.get(term, float32(-np.inf))
            if score > old:
                score_map[term] = score
                queue.append(term)

        qh = 0
        while qh < len(queue):
            term = queue[qh]
            qh += 1
            score = score_map.get(term, float32(-np.inf))
            ps = int(parents_indptr[int(term)])
            pe = int(parents_indptr[int(term) + 1])
            for p in range(ps, pe):
                parent = int32(parents_indices[p])
                old = score_map.get(parent, float32(-np.inf))
                if score > old:
                    score_map[parent] = score
                    queue.append(parent)

        m = len(score_map)
        tmp_terms = np.empty(m, dtype=np.int32)
        tmp_scores = np.empty(m, dtype=np.float32)
        t = 0
        for term, score in score_map.items():
            tmp_terms[t] = int32(term)
            tmp_scores[t] = float32(score)
            t += 1

        order = np.argsort(tmp_terms)
        out_pos = int(out_indptr[row_index])
        for j in range(m):
            k = int(order[j])
            out_indices[out_pos + j] = int32(tmp_terms[k])
            out_scores[out_pos + j] = float32(tmp_scores[k])

    return out_indptr, out_indices, out_scores


@nb.njit(cache=True)
def filter_terms_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    scores: np.ndarray,
    ia: np.ndarray,
    train_counts: np.ndarray,
    weak_min_train_n: int,
    weak_score_thr: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter terms by IA and train counts + threshold as CSR."""
    n_rows = int(indptr.size - 1)
    out_counts = np.zeros(n_rows, dtype=np.int64)

    for row_index in range(n_rows):
        kept = 0
        start = int(indptr[row_index])
        end = int(indptr[row_index + 1])
        for pos in range(start, end):
            term = indices[pos]
            score = scores[pos]

            if ia[term] <= 0.0:
                continue
            if train_counts[term] < weak_min_train_n and score < weak_score_thr:
                continue

            kept += 1
        out_counts[row_index] = kept

    out_indptr = np.empty(n_rows + 1, dtype=np.int64)
    out_indptr[0] = 0
    for i in range(n_rows):
        out_indptr[i + 1] = out_indptr[i] + out_counts[i]

    total = int(out_indptr[n_rows])
    out_indices = np.empty(total, dtype=np.int32)
    out_scores = np.empty(total, dtype=np.float32)

    for row_index in range(n_rows):
        out_pos = int(out_indptr[row_index])
        start = int(indptr[row_index])
        end = int(indptr[row_index + 1])
        for pos in range(start, end):
            term = indices[pos]
            score = scores[pos]

            if ia[term] <= 0.0:
                continue
            if train_counts[term] < weak_min_train_n and score < weak_score_thr:
                continue

            out_indices[out_pos] = term
            out_scores[out_pos] = score
            out_pos += 1

    return out_indptr, out_indices, out_scores


@nb.njit(cache=True)
def topk_from_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    scores: np.ndarray,
    top_k: int,
    max_k: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select up to max_k terms per row: all terms with score > cut plus
    as many terms with score == cut as fit into max_k,
    where cut is the top_k-th highest score. Pads with -1 / -inf.
    """
    n_rows = int(indptr.size - 1)
    topk_pos = np.full((n_rows, int(max_k)), -1, dtype=np.int32)
    topk_scores = np.full((n_rows, int(max_k)), float32(-np.inf), dtype=np.float32)

    for row_index in range(n_rows):
        start = int(indptr[row_index])
        end = int(indptr[row_index + 1])
        m = end - start
        if m <= 0:
            continue

        if m <= int(top_k):
            k = m if m <= int(max_k) else int(max_k)
            row_scores = scores[start:end]
            idx = np.argsort(row_scores)[::-1]
            for j in range(k):
                p = start + int(idx[j])
                topk_pos[row_index, j] = int32(indices[p])
                topk_scores[row_index, j] = float32(scores[p])
            continue

        k = int(top_k)
        row_scores = scores[start:end]
        kth_pos = m - k
        part = np.argpartition(row_scores, kth_pos)
        cut = float32(row_scores[int(part[kth_pos])])

        tmp_pos = np.empty(m, dtype=np.int32)
        tmp_sc = np.empty(m, dtype=np.float32)
        t = 0

        for p in range(m):
            s = float32(row_scores[p])
            if s > cut:
                tmp_pos[t] = np.int32(p)
                tmp_sc[t] = s
                t += 1

        for p in range(m):
            if t >= int(max_k):
                break
            s = float32(row_scores[p])
            if s == cut:
                tmp_pos[t] = np.int32(p)
                tmp_sc[t] = s
                t += 1

        if t <= 0:
            continue

        ord_idx = np.argsort(tmp_sc[:t])[::-1]
        for j in range(t):
            p = start + int(tmp_pos[int(ord_idx[j])])
            topk_pos[row_index, j] = int32(indices[p])
            topk_scores[row_index, j] = float32(scores[p])

    return topk_pos, topk_scores


@nb.njit(cache=True)
def select_topk_state_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    scores: np.ndarray,
    top_k: int,
    max_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = int(indptr.size - 1)
    effective_k = int(top_k)
    max_k_int = int(max_k)
    if effective_k > max_k_int:
        effective_k = max_k_int

    out_counts = np.zeros(n_rows, dtype=np.int64)

    for row_index in range(n_rows):
        start = int(indptr[row_index])
        end = int(indptr[row_index + 1])
        row_nnz = end - start
        if row_nnz <= 0:
            continue
        if row_nnz <= effective_k:
            out_counts[row_index] = row_nnz
            continue

        row_scores = scores[start:end]
        kth_pos = row_nnz - effective_k
        part = np.argpartition(row_scores, kth_pos)
        cut = float32(row_scores[int(part[kth_pos])])

        n_gt = 0
        for pos in range(row_nnz):
            if float32(row_scores[pos]) > cut:
                n_gt += 1

        keep = n_gt
        if keep < max_k_int:
            for pos in range(row_nnz):
                if keep >= max_k_int:
                    break
                if float32(row_scores[pos]) == cut:
                    keep += 1

        out_counts[row_index] = keep

    out_indptr = np.empty(n_rows + 1, dtype=np.int64)
    out_indptr[0] = 0
    for row_index in range(n_rows):
        out_indptr[row_index + 1] = out_indptr[row_index] + out_counts[row_index]

    total = int(out_indptr[n_rows])
    out_indices = np.empty(total, dtype=np.int32)
    out_scores = np.empty(total, dtype=np.float32)

    for row_index in range(n_rows):
        start = int(indptr[row_index])
        end = int(indptr[row_index + 1])
        row_nnz = end - start
        if row_nnz <= 0:
            continue

        out_pos = int(out_indptr[row_index])
        out_end = int(out_indptr[row_index + 1])
        if out_pos == out_end:
            continue

        if row_nnz <= effective_k:
            out_indices[out_pos:out_end] = indices[start:end]
            out_scores[out_pos:out_end] = scores[start:end]
            continue

        row_scores = scores[start:end]
        kth_pos = row_nnz - effective_k
        part = np.argpartition(row_scores, kth_pos)
        cut = float32(row_scores[int(part[kth_pos])])

        n_gt = 0
        for pos in range(row_nnz):
            if float32(row_scores[pos]) > cut:
                n_gt += 1
        tie_left = max_k_int - n_gt

        for pos in range(row_nnz):
            if out_pos >= out_end:
                break
            sc = float32(row_scores[pos])
            if sc > cut:
                out_indices[out_pos] = int32(indices[start + pos])
                out_scores[out_pos] = float32(scores[start + pos])
                out_pos += 1
            elif sc == cut and tie_left > 0:
                out_indices[out_pos] = int32(indices[start + pos])
                out_scores[out_pos] = float32(scores[start + pos])
                out_pos += 1
                tie_left -= 1

    return out_indptr, out_indices, out_scores
