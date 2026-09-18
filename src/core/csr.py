"""
CSR utilities for building and manipulating CSR matrices.
"""
import numpy as np
import numba as nb

def empty_csr(n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Return empty CSR (indptr, indices) with n_rows rows."""
    indptr = np.zeros(int(n_rows) + 1, dtype=np.int64)
    return indptr, np.empty(0, dtype=np.int32)


def pack_csr_from_pairs(
    row_index: np.ndarray,
    col_index: np.ndarray,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pack CSR from (row_index, col_index) pairs.

    Contract:
    - row_index, col_index are 1D and same length
    - 0 <= row_index < n_rows
    - col_index >= 0
    """
    row_index = np.asarray(row_index, dtype=np.int64)
    col_index = np.asarray(col_index, dtype=np.int32)
    n_rows = int(n_rows)

    if row_index.size == 0 or n_rows == 0:
        return empty_csr(n_rows)

    order = np.lexsort((col_index, row_index))
    row_sorted = row_index[order]
    col_sorted = col_index[order]

    if row_sorted.size > 1:
        keep = np.ones(row_sorted.size, dtype=bool)
        keep[1:] = ~((row_sorted[1:] == row_sorted[:-1]) &
                    (col_sorted[1:] == col_sorted[:-1]))
        row_sorted = row_sorted[keep]
        col_sorted = col_sorted[keep]

    counts = np.bincount(row_sorted, minlength=n_rows).astype(np.int64, copy=False)
    indptr = np.empty(n_rows + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])

    return indptr, col_sorted.astype(np.int32, copy=False)


def invert_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    n_cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Invert CSR adjacency: rows->cols becomes cols->rows.

    Contract:
    - indptr is valid CSR: indptr[0]=0, indptr[-1]=len(indices)
    - 0 <= indices < n_cols
    """
    indptr = np.asarray(indptr, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int32)
    n_cols = int(n_cols)

    if indices.size == 0 or n_cols == 0:
        return empty_csr(n_cols)

    n_rows = int(indptr.size - 1)
    col_counts = np.bincount(indices.astype(np.int64, copy=False),
                             minlength=n_cols).astype(np.int64, copy=False)

    indptr_t = np.empty(n_cols + 1, dtype=np.int64)
    indptr_t[0] = 0
    np.cumsum(col_counts, out=indptr_t[1:])

    indices_t = np.empty(indices.size, dtype=np.int32)
    write_ptr = indptr_t[:-1].copy()

    for row in range(n_rows):
        start = int(indptr[row])
        end = int(indptr[row + 1])
        for pos in range(start, end):
            col = int(indices[pos])
            out_pos = int(write_ptr[col])
            indices_t[out_pos] = np.int32(row)
            write_ptr[col] = out_pos + 1

    return indptr_t, indices_t


def pack_row_aligned_csr(
    row_ids_sorted: np.ndarray,
    entry_ids: np.ndarray,
    col_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build CSR over rows=row_ids_sorted from (entry_ids, col_index) pairs.

    Contract:
    - row_ids_sorted is sorted unique (dtype object)
    - entry_ids and col_index are 1D same length
    - col_index >= -1 (use -1 for "drop")
    """
    row_ids_sorted = np.asarray(row_ids_sorted, dtype=object)
    entry_ids = np.asarray(entry_ids, dtype=object)
    col_index = np.asarray(col_index, dtype=np.int32)

    if row_ids_sorted.size == 0:
        return empty_csr(0)
    if entry_ids.size == 0:
        return empty_csr(int(row_ids_sorted.size))

    pos = np.searchsorted(row_ids_sorted, entry_ids, side="left").astype(np.int64)
    in_bounds = pos < int(row_ids_sorted.size)
    matched = np.zeros(entry_ids.size, dtype=bool)
    matched[in_bounds] = row_ids_sorted[pos[in_bounds]] == entry_ids[in_bounds]

    keep = matched & (col_index >= 0)
    if not np.any(keep):
        return empty_csr(int(row_ids_sorted.size))

    return pack_csr_from_pairs(
        row_index=pos[keep],
        col_index=col_index[keep],
        n_rows=int(row_ids_sorted.size),
    )


def pack_csr_from_pairs_with_data(
    row_index: np.ndarray,
    col_index: np.ndarray,
    data: np.ndarray,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pack CSR from (row_index, col_index, data) triples.

    Contract:
    - row_index, col_index, data are 1D and same length
    - 0 <= row_index < n_rows
    - col_index >= 0
    - (row_index, col_index) pairs are unique
    """
    row_index = np.asarray(row_index, dtype=np.int64)
    col_index = np.asarray(col_index, dtype=np.int32)
    data = np.asarray(data)
    n_rows = int(n_rows)

    if row_index.size == 0 or n_rows == 0:
        indptr, indices = empty_csr(n_rows)
        return indptr, indices, np.empty(0, dtype=data.dtype)

    order = np.lexsort((col_index, row_index))
    row_sorted = row_index[order]
    col_sorted = col_index[order]
    data_sorted = data[order]

    counts = np.bincount(row_sorted, minlength=n_rows).astype(np.int64, copy=False)
    indptr = np.empty(n_rows + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])

    return indptr, col_sorted.astype(np.int32, copy=False), data_sorted


def pack_row_aligned_csr_with_data(
    row_ids_sorted: np.ndarray,
    entry_ids: np.ndarray,
    col_index: np.ndarray,
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build CSR (indptr, indices, data) over rows=row_ids_sorted.

    Contract:
    - row_ids_sorted is sorted unique (dtype object)
    - entry_ids, col_index, data are 1D same length
    - col_index >= -1 (use -1 for "drop")
    - (row_index, col_index) pairs are unique
    """
    row_ids_sorted = np.asarray(row_ids_sorted, dtype=object)
    entry_ids = np.asarray(entry_ids, dtype=object)
    col_index = np.asarray(col_index, dtype=np.int32)
    data = np.asarray(data)

    if row_ids_sorted.size == 0:
        indptr, indices = empty_csr(0)
        return indptr, indices, np.empty(0, dtype=data.dtype)
    if entry_ids.size == 0:
        indptr, indices = empty_csr(int(row_ids_sorted.size))
        return indptr, indices, np.empty(0, dtype=data.dtype)

    pos = np.searchsorted(row_ids_sorted, entry_ids, side="left").astype(np.int64)
    in_bounds = pos < int(row_ids_sorted.size)
    matched = np.zeros(entry_ids.size, dtype=bool)
    matched[in_bounds] = row_ids_sorted[pos[in_bounds]] == entry_ids[in_bounds]

    keep = matched & (col_index >= 0)
    if not np.any(keep):
        indptr, indices = empty_csr(int(row_ids_sorted.size))
        return indptr, indices, np.empty(0, dtype=data.dtype)

    return pack_csr_from_pairs_with_data(
        row_index=pos[keep],
        col_index=col_index[keep],
        data=data[keep],
        n_rows=int(row_ids_sorted.size),
    )


@nb.njit(cache=True)
def csr_from_dense(
    term_pos: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a CSR (sorted by term_pos within each row) from dense top-k arrays.

    Contract:
    - term_pos is int32 array of shape (n_rows, k_max).
    - scores is float32 array of shape (n_rows, k_max).
    - Padding is encoded as term_pos == -1 and/or scores == -inf.
    - All valid term_pos values satisfy 0 <= term_pos < 2^31.
    - Output indices are sorted ascending within each row and contain no padding.
    - Output indptr is int64 with indptr[0] == 0 and indptr[-1] == len(indices).
    """
    n_rows = int(term_pos.shape[0])
    k_max = int(term_pos.shape[1])
    neg_inf = np.float32(-np.inf)

    counts = np.zeros(n_rows, dtype=np.int64)
    for r in range(n_rows):
        c = 0
        for j in range(k_max):
            term = term_pos[r, j]
            if term < 0:
                continue
            sc = scores[r, j]
            if sc == neg_inf:
                continue
            c += 1
        counts[r] = c

    indptr = np.empty(n_rows + 1, dtype=np.int64)
    indptr[0] = 0
    for r in range(n_rows):
        indptr[r + 1] = indptr[r] + counts[r]

    nnz = int(indptr[n_rows])
    indices = np.empty(nnz, dtype=np.int32)
    out_scores = np.empty(nnz, dtype=np.float32)

    tmp_terms = np.empty(k_max, dtype=np.int32)
    tmp_scores = np.empty(k_max, dtype=np.float32)

    for r in range(n_rows):
        m = int(counts[r])
        if m == 0:
            continue

        t = 0
        for j in range(k_max):
            term = term_pos[r, j]
            if term < 0:
                continue
            sc = scores[r, j]
            if sc == neg_inf:
                continue
            tmp_terms[t] = term
            tmp_scores[t] = sc
            t += 1

        order = np.argsort(tmp_terms[:m])
        start = int(indptr[r])
        for j in range(m):
            p = int(order[j])
            indices[start + j] = tmp_terms[p]
            out_scores[start + j] = tmp_scores[p]

    return indptr, indices, out_scores
