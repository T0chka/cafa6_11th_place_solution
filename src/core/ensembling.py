"""Ensembling functions for CSRs."""

import numpy as np
import numba as nb
from numba import int32, float32



@nb.njit(cache=True)
def mean_inplace(sum_scores: np.ndarray, sum_counts: np.ndarray) -> None:
    n = int(sum_scores.size)
    for i in range(n):
        sum_scores[i] = float32(sum_scores[i]) / float32(sum_counts[i])


@nb.njit(cache=True)
def merge_max_csr(
    indptr_a: np.ndarray,
    indices_a: np.ndarray,
    scores_a: np.ndarray,
    indptr_b: np.ndarray,
    indices_b: np.ndarray,
    scores_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge two CSRs by max score."""
    n_rows = int(indptr_a.size - 1)
    out_counts = np.zeros(n_rows, dtype=np.int64)

    for row_index in range(n_rows):
        a0 = int(indptr_a[row_index])
        a1 = int(indptr_a[row_index + 1])
        b0 = int(indptr_b[row_index])
        b1 = int(indptr_b[row_index + 1])
        i = a0
        j = b0
        kept = 0
        while i < a1 and j < b1:
            ta = int(indices_a[i])
            tb = int(indices_b[j])
            if ta == tb:
                kept += 1
                i += 1
                j += 1
            elif ta < tb:
                kept += 1
                i += 1
            else:
                kept += 1
                j += 1
        kept += (a1 - i) + (b1 - j)
        out_counts[row_index] = kept

    out_indptr = np.empty(n_rows + 1, dtype=np.int64)
    out_indptr[0] = 0
    for i in range(n_rows):
        out_indptr[i + 1] = out_indptr[i] + out_counts[i]

    total = int(out_indptr[n_rows])
    out_indices = np.empty(total, dtype=np.int32)
    out_scores = np.empty(total, dtype=np.float32)

    for row_index in range(n_rows):
        a0 = int(indptr_a[row_index])
        a1 = int(indptr_a[row_index + 1])
        b0 = int(indptr_b[row_index])
        b1 = int(indptr_b[row_index + 1])
        i = a0
        j = b0
        out_pos = int(out_indptr[row_index])

        while i < a1 and j < b1:
            ta = int(indices_a[i])
            tb = int(indices_b[j])
            if ta == tb:
                sa = float32(scores_a[i])
                sb = float32(scores_b[j])
                out_indices[out_pos] = int32(ta)
                out_scores[out_pos] = sa if sa >= sb else sb
                out_pos += 1
                i += 1
                j += 1
            elif ta < tb:
                out_indices[out_pos] = int32(ta)
                out_scores[out_pos] = float32(scores_a[i])
                out_pos += 1
                i += 1
            else:
                out_indices[out_pos] = int32(tb)
                out_scores[out_pos] = float32(scores_b[j])
                out_pos += 1
                j += 1

        while i < a1:
            ta = int(indices_a[i])
            out_indices[out_pos] = int32(ta)
            out_scores[out_pos] = float32(scores_a[i])
            out_pos += 1
            i += 1

        while j < b1:
            tb = int(indices_b[j])
            out_indices[out_pos] = int32(tb)
            out_scores[out_pos] = float32(scores_b[j])
            out_pos += 1
            j += 1

    return out_indptr, out_indices, out_scores


@nb.njit(cache=True)
def merge_sum_count_csr(
    indptr_sum: np.ndarray,
    indices_sum: np.ndarray,
    sum_scores: np.ndarray,
    sum_counts: np.ndarray,
    indptr_x: np.ndarray,
    indices_x: np.ndarray,
    scores_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge two CSRs by sum score and count."""
    n_rows = int(indptr_sum.size - 1)
    out_counts = np.zeros(n_rows, dtype=np.int64)

    for row_index in range(n_rows):
        a0 = int(indptr_sum[row_index])
        a1 = int(indptr_sum[row_index + 1])
        b0 = int(indptr_x[row_index])
        b1 = int(indptr_x[row_index + 1])
        i = a0
        j = b0
        kept = 0
        while i < a1 and j < b1:
            ta = int(indices_sum[i])
            tb = int(indices_x[j])
            if ta == tb:
                kept += 1
                i += 1
                j += 1
            elif ta < tb:
                kept += 1
                i += 1
            else:
                kept += 1
                j += 1
        kept += (a1 - i) + (b1 - j)
        out_counts[row_index] = kept

    out_indptr = np.empty(n_rows + 1, dtype=np.int64)
    out_indptr[0] = 0
    for i in range(n_rows):
        out_indptr[i + 1] = out_indptr[i] + out_counts[i]

    total = int(out_indptr[n_rows])
    out_indices = np.empty(total, dtype=np.int32)
    out_sum = np.empty(total, dtype=np.float32)
    out_cnt = np.empty(total, dtype=np.int16)

    for row_index in range(n_rows):
        a0 = int(indptr_sum[row_index])
        a1 = int(indptr_sum[row_index + 1])
        b0 = int(indptr_x[row_index])
        b1 = int(indptr_x[row_index + 1])
        i = a0
        j = b0
        out_pos = int(out_indptr[row_index])

        while i < a1 and j < b1:
            ta = int(indices_sum[i])
            tb = int(indices_x[j])
            if ta == tb:
                out_indices[out_pos] = int32(ta)
                out_sum[out_pos] = float32(sum_scores[i]) + float32(scores_x[j])
                out_cnt[out_pos] = np.int16(sum_counts[i] + 1)
                out_pos += 1
                i += 1
                j += 1
            elif ta < tb:
                out_indices[out_pos] = int32(ta)
                out_sum[out_pos] = float32(sum_scores[i])
                out_cnt[out_pos] = np.int16(sum_counts[i])
                out_pos += 1
                i += 1
            else:
                out_indices[out_pos] = int32(tb)
                out_sum[out_pos] = float32(scores_x[j])
                out_cnt[out_pos] = np.int16(1)
                out_pos += 1
                j += 1

        while i < a1:
            ta = int(indices_sum[i])
            out_indices[out_pos] = int32(ta)
            out_sum[out_pos] = float32(sum_scores[i])
            out_cnt[out_pos] = np.int16(sum_counts[i])
            out_pos += 1
            i += 1

        while j < b1:
            tb = int(indices_x[j])
            out_indices[out_pos] = int32(tb)
            out_sum[out_pos] = float32(scores_x[j])
            out_cnt[out_pos] = np.int16(1)
            out_pos += 1
            j += 1

    return out_indptr, out_indices, out_sum, out_cnt
