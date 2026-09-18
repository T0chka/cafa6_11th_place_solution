"""
Scoring functions (adapted from CAFA-evaluator-PK)
"""

import pandas as pd
import numpy as np
import numba as nb


def _compute_f(pr: np.ndarray, rc: np.ndarray) -> np.ndarray:
    r"""Compute the F-score from precision and recall."""
    n = 2.0 * pr * rc
    d = pr + rc
    return np.divide(n, d, out=np.zeros_like(n, dtype=float), where=d != 0.0)


def _compute_s(ru: np.ndarray, mi: np.ndarray) -> np.ndarray:
    r"""Compute the S-metric from RU and MI."""
    return np.sqrt(ru * ru + mi * mi)


def _normalize(
    raw: pd.DataFrame,
    aspect: str,
    tau_arr: np.ndarray,
    ne: np.ndarray,
    normalization: str = "cafa",
) -> pd.DataFrame:
    r"""Apply CAFA-style normalization and derived metrics."""
    metrics = raw.copy()

    for col in ("tp", "fp", "fn", "pr", "rc"):
        denom = ne
        if normalization == "pred" or (normalization == "cafa" and col == "pr"):
            denom = metrics["n"].to_numpy(dtype=float)
        metrics[col] = np.divide(
            metrics[col].to_numpy(dtype=float),
            denom,
            out=np.zeros_like(metrics[col].to_numpy(dtype=float)),
            where=np.asarray(denom, dtype=float) > 0.0,
        )

    metrics["aspect"] = aspect
    metrics["tau"] = tau_arr
    metrics["cov"] = metrics["n"].to_numpy(dtype=float) / ne
    metrics["mi"] = metrics["fp"]
    metrics["ru"] = metrics["fn"]
    metrics["f"] = _compute_f(metrics["pr"].to_numpy(), metrics["rc"].to_numpy())
    metrics["s"] = _compute_s(metrics["ru"].to_numpy(), metrics["mi"].to_numpy())

    tp = metrics["tp"].to_numpy(dtype=float)
    fp = metrics["fp"].to_numpy(dtype=float)
    fn = metrics["fn"].to_numpy(dtype=float)

    metrics["pr_micro"] = np.divide(
        tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0.0
    )
    metrics["rc_micro"] = np.divide(
        tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0.0
    )
    metrics["f_micro"] = _compute_f(
        metrics["pr_micro"].to_numpy(),
        metrics["rc_micro"].to_numpy()
    )
    metrics["tp_in_pred"] = metrics["pr_micro"]
    metrics["tp_missed"] = 1.0 - metrics["rc_micro"]
    metrics["fp_in_pred"] = 1.0 - metrics["pr_micro"]

    return metrics


@nb.njit(cache=True)
def _binsearch_int32(a: np.ndarray, lo: int, hi: int, x: int) -> int:
    left = lo
    right = hi - 1
    while left <= right:
        mid = (left + right) >> 1
        v = int(a[mid])
        if v < x:
            left = mid + 1
        elif v > x:
            right = mid - 1
        else:
            return 1
    return 0


@nb.njit(cache=True)
def _gt_weights_from_csr(
    gt_indptr: np.ndarray,
    gt_indices: np.ndarray,
    toi_pos: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    n_rows = int(gt_indptr.size - 1)
    out = np.zeros(n_rows, dtype=np.float32)
    for i in range(n_rows):
        s = int(gt_indptr[i])
        e = int(gt_indptr[i + 1])
        acc = np.float32(0.0)
        for k in range(s, e):
            t = int(gt_indices[k])
            p = int(toi_pos[t])
            if p >= 0:
                acc += np.float32(weights[p])
        out[i] = acc
    return out


@nb.njit(cache=False)
def _compute_metrics_csr(
    tau_arr: np.ndarray,
    pred_terms: np.ndarray,
    pred_scores: np.ndarray,
    gt_indptr: np.ndarray,
    gt_indices: np.ndarray,
    weights: np.ndarray,
    toi_pos: np.ndarray,
    gt_w: np.ndarray,
) -> np.ndarray:
    n_tau = int(tau_arr.size)
    n_rows = int(pred_terms.shape[0])
    top_k = int(pred_terms.shape[1])
    out = np.zeros((n_tau, 6), dtype=np.float64)

    for i in range(n_rows):
        row_s = int(gt_indptr[i])
        row_e = int(gt_indptr[i + 1])
        gw = np.float32(gt_w[i])

        w_pred = np.zeros(top_k, dtype=np.float32)
        hit = np.zeros(top_k, dtype=np.uint8)
        for j in range(top_k):
            t = int(pred_terms[i, j])
            if t < 0:
                continue
            p = int(toi_pos[t])
            if p < 0:
                continue
            w_pred[j] = np.float32(weights[p])
            if _binsearch_int32(gt_indices, row_s, row_e, t) == 1:
                hit[j] = 1

        ptr = 0
        pred_w = np.float32(0.0)
        tp_w = np.float32(0.0)
        scores_row = pred_scores[i, :]

        for ti in range(n_tau - 1, -1, -1):
            tau = tau_arr[ti]
            while ptr < top_k and scores_row[ptr] >= tau:
                w = w_pred[ptr]
                if w != 0.0:
                    pred_w += w
                    if hit[ptr] != 0:
                        tp_w += w
                ptr += 1

            if pred_w > 0.0:
                out[ti, 0] += 1.0
                out[ti, 4] += np.float64(tp_w / pred_w)

            if gw > 0.0:
                out[ti, 5] += np.float64(tp_w / gw)

            out[ti, 1] += np.float64(tp_w)
            out[ti, 2] += np.float64(pred_w - tp_w)
            out[ti, 3] += np.float64(gw - tp_w)

    return out


def _align_preds_to_gt(
    gt_ids: np.ndarray,
    pred_entry_ids: np.ndarray,
    pred_terms: np.ndarray,
    pred_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gt_ids = gt_ids.astype(object, copy=False)
    pred_entry_ids = pred_entry_ids.astype(object, copy=False)
    pred_terms = pred_terms.astype(np.int32, copy=False)
    pred_scores = pred_scores.astype(np.float32, copy=False)

    n_gt = int(gt_ids.size)
    top_k = int(pred_terms.shape[1])

    aligned_terms = np.full((n_gt, top_k), -1, dtype=np.int32)
    aligned_scores = np.zeros((n_gt, top_k), dtype=np.float32)

    id_to_pred_row: dict[object, int] = {}
    for row in range(int(pred_entry_ids.size)):
        entry_id = pred_entry_ids[row]
        if entry_id not in id_to_pred_row:
            id_to_pred_row[entry_id] = row

    for gt_row in range(n_gt):
        entry_id = gt_ids[gt_row]
        pred_row = id_to_pred_row.get(entry_id)
        if pred_row is None:
            continue
        aligned_terms[gt_row, :] = pred_terms[pred_row, :]
        aligned_scores[gt_row, :] = pred_scores[pred_row, :]

    return aligned_terms, aligned_scores


@nb.njit(cache=True)
def _csr_diff_sorted(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(a.size, dtype=np.int32)
    i = 0
    j = 0
    k = 0
    na = int(a.size)
    nb_ = int(b.size)
    while i < na and j < nb_:
        va = int(a[i])
        vb = int(b[j])
        if va < vb:
            out[k] = np.int32(va)
            k += 1
            i += 1
        elif va > vb:
            j += 1
        else:
            i += 1
            j += 1
    while i < na:
        out[k] = np.int32(a[i])
        k += 1
        i += 1
    return out[:k]


def eval_gt_minus_known_aligned(aspect_gt: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build evaluation GT as gt \\ known_terms, aligned by EntryID."""
    gt = aspect_gt["gt"]
    gt_ids = gt["gt_ids"]
    gt_indptr = gt["gt_indptr"]
    gt_indices = gt["gt_indices"]

    known = aspect_gt["known_terms"]
    known_ids = known["known_ids"]
    known_indptr = known["known_indptr"]
    known_indices = known["known_indices"]

    n_gt = int(gt_ids.size)
    n_known = int(known_ids.size)

    out_indptr = np.empty(n_gt + 1, dtype=np.int64)
    out_indptr[0] = 0
    parts: list[np.ndarray] = []

    j = 0
    for i in range(n_gt):
        gt_id = gt_ids[i]
        while j < n_known and known_ids[j] < gt_id:
            j += 1

        gs = int(gt_indptr[i])
        ge = int(gt_indptr[i + 1])

        if j < n_known and known_ids[j] == gt_id:
            ks = int(known_indptr[j])
            ke = int(known_indptr[j + 1])
            diff = _csr_diff_sorted(gt_indices[gs:ge], known_indices[ks:ke])
        else:
            diff = gt_indices[gs:ge]

        parts.append(diff)
        out_indptr[i + 1] = out_indptr[i] + int(diff.size)

    out_indices = np.concatenate(parts) if parts else np.empty(0, np.int32)
    return gt_ids, out_indptr, out_indices


def score_weighted_f(
    prepared_gt: dict[str, dict],
    pred_terms_by_aspect: dict[str, np.ndarray],
    pred_scores_by_aspect: dict[str, np.ndarray],
    th_step: float = 0.01,
    eval_on: str = "gt", # "withheld"
) -> pd.DataFrame:
    tau_arr = np.arange(th_step, 1.0, th_step, dtype=float)

    dfs: list[pd.DataFrame] = []

    # Get set of aspects that have predictions
    aspects_with_preds = set(pred_terms_by_aspect.keys()) & set(pred_scores_by_aspect.keys())
    if not aspects_with_preds:
        return pd.DataFrame(columns=[
            "aspect", "tau", "n", "tp", "fp", "fn", "pr", "rc", "cov", "mi", "ru",
            "f", "s", "pr_micro", "rc_micro", "f_micro",
        ])

    # Process only aspects that have both predictions and ground truth
    for aspect in aspects_with_preds:
        if aspect not in prepared_gt:
            continue

        aspect_gt = prepared_gt[aspect]
        pred_entry_ids, pred_terms = pred_terms_by_aspect[aspect]
        pred_scores = pred_scores_by_aspect[aspect]

        if eval_on == "withheld":
            gt_ids, gt_indptr, gt_indices = eval_gt_minus_known_aligned(aspect_gt)
        else:
            gt = aspect_gt["gt"]
            gt_ids = gt["gt_ids"]
            gt_indptr = gt["gt_indptr"]
            gt_indices = gt["gt_indices"]

        pred_terms, pred_scores = _align_preds_to_gt(
            gt_ids=gt_ids,
            pred_entry_ids=pred_entry_ids,
            pred_terms=pred_terms,
            pred_scores=pred_scores,
        )

        ia = aspect_gt["ia"]
        toi = np.flatnonzero(ia > 0.0).astype(np.int32, copy=False)
        weights = ia[toi].astype(np.float32, copy=False)

        n_terms = int(ia.size)
        toi_pos = np.full(n_terms, -1, dtype=np.int32)
        for j in range(int(toi.size)):
            toi_pos[int(toi[j])] = np.int32(j)

        gt_w_full = _gt_weights_from_csr(
            gt_indptr.astype(np.int64, copy=False),
            gt_indices.astype(np.int32, copy=False),
            toi_pos.astype(np.int32, copy=False),
            weights.astype(np.float32, copy=False),
        )

        idx = np.flatnonzero(gt_w_full > np.float32(0.0)).astype(np.int64, copy=False)
        if idx.size == 0:
            continue

        pred_terms = pred_terms[idx, :]
        pred_scores = pred_scores[idx, :]
        gt_w = gt_w_full[idx].astype(np.float32, copy=False)

        gt_indptr_full = gt_indptr.astype(np.int64, copy=False)
        gt_indices_full = gt_indices.astype(np.int32, copy=False)

        gt_indptr_sub = np.empty(idx.size + 1, dtype=np.int64)
        gt_indptr_sub[0] = 0
        for i in range(int(idx.size)):
            row = int(idx[i])
            gt_indptr_sub[i + 1] = gt_indptr_sub[i] + (
                gt_indptr_full[row + 1] - gt_indptr_full[row]
            )

        nnz = int(gt_indptr_sub[-1])
        gt_indices_sub = np.empty(nnz, dtype=np.int32)
        pos = 0
        for i in range(int(idx.size)):
            row = int(idx[i])
            s = int(gt_indptr_full[row])
            e = int(gt_indptr_full[row + 1])
            m = e - s
            gt_indices_sub[pos:pos + m] = gt_indices_full[s:e]
            pos += m

        metrics_arr = _compute_metrics_csr(
            tau_arr,
            pred_terms.astype(np.int32, copy=False),
            pred_scores.astype(np.float32, copy=False),
            gt_indptr_sub.astype(np.int64, copy=False),
            gt_indices_sub.astype(np.int32, copy=False),
            weights.astype(np.float32, copy=False),
            toi_pos.astype(np.int32, copy=False),
            gt_w.astype(np.float32, copy=False),
        )

        raw = pd.DataFrame(metrics_arr, columns=["n", "tp", "fp", "fn", "pr", "rc"])
        ne_scalar = int(pred_terms.shape[0])
        ne = np.full(tau_arr.size, ne_scalar, dtype=float)
        dfs.append(_normalize(raw, aspect, tau_arr, ne))

    if not dfs:
        return pd.DataFrame(columns=[
            "aspect", "tau", "n", "tp", "fp", "fn", "pr", "rc", "cov", "mi", "ru",
            "f", "s", "pr_micro", "rc_micro", "f_micro", "tp_in_pred", "tp_missed", "fp_in_pred",
        ])

    return pd.concat(dfs, ignore_index=True)


@nb.njit(cache=False)
def _weighted_ndcg_at_k_csr(
    pred_terms: np.ndarray,
    gt_indptr: np.ndarray,
    gt_indices: np.ndarray,
    ia: np.ndarray,
    discounts: np.ndarray,
    k: int,
) -> tuple[np.float64, np.int64]:
    n_rows = int(pred_terms.shape[0])
    top_k = int(pred_terms.shape[1])
    k_eff = int(k) if int(k) < int(top_k) else int(top_k)

    ndcg_sum = np.float64(0.0)
    n_eval = np.int64(0)

    for i in range(n_rows):
        gs = int(gt_indptr[i])
        ge = int(gt_indptr[i + 1])
        if ge <= gs:
            continue

        m = ge - gs
        gains = np.empty(m, dtype=np.float32)
        g = 0
        for p in range(gs, ge):
            t = int(gt_indices[p])
            w = np.float32(ia[t])
            if w > np.float32(0.0):
                gains[g] = w
                g += 1
        if g == 0:
            continue

        gains = gains[:g]
        gains.sort()

        idcg = np.float64(0.0)
        lim = int(g) if int(g) < int(k_eff) else int(k_eff)
        for r in range(lim):
            idcg += np.float64(gains[g - 1 - r]) * np.float64(discounts[r])
        if idcg <= np.float64(0.0):
            continue

        dcg = np.float64(0.0)
        for r in range(k_eff):
            t = int(pred_terms[i, r])
            if t < 0:
                continue
            w = np.float32(ia[t])
            if w <= np.float32(0.0):
                continue
            if _binsearch_int32(gt_indices, gs, ge, t) == 1:
                dcg += np.float64(w) * np.float64(discounts[r])

        ndcg_sum += dcg / idcg
        n_eval += 1

    return ndcg_sum, n_eval


def score_weighted_ndcg(
    prepared_gt: dict[str, dict],
    pred_terms_by_aspect: dict[str, tuple[np.ndarray, np.ndarray]],
    pred_scores_by_aspect: dict[str, np.ndarray],
    eval_on: str = "gt",
    ndcg_k: int | None = None,
) -> pd.DataFrame:
    aspects = set(pred_terms_by_aspect.keys()) & set(pred_scores_by_aspect.keys())
    if not aspects:
        return pd.DataFrame(columns=["aspect", "ndcg", "n_eval"])

    out_rows: list[tuple[str, float, int]] = []

    for aspect in aspects:
        if aspect not in prepared_gt:
            continue

        aspect_gt = prepared_gt[aspect]
        pred_entry_ids, pred_terms = pred_terms_by_aspect[aspect]
        pred_scores = pred_scores_by_aspect[aspect]

        if eval_on == "withheld":
            gt_ids, gt_indptr, gt_indices = eval_gt_minus_known_aligned(aspect_gt)
        else:
            gt = aspect_gt["gt"]
            gt_ids = gt["gt_ids"]
            gt_indptr = gt["gt_indptr"]
            gt_indices = gt["gt_indices"]

        aligned_terms, _ = _align_preds_to_gt(
            gt_ids=gt_ids,
            pred_entry_ids=pred_entry_ids,
            pred_terms=pred_terms,
            pred_scores=pred_scores,
        )

        top_k = int(aligned_terms.shape[1])
        k_eval = top_k if ndcg_k is None else int(ndcg_k)
        k_eval = max(0, min(top_k, k_eval))

        if k_eval == 0:
            out_rows.append((str(aspect), 0.0, 0))
            continue

        discounts = (1.0 / np.log2(np.arange(2, k_eval + 2))).astype(np.float32)

        ndcg_sum, n_eval = _weighted_ndcg_at_k_csr(
            pred_terms=aligned_terms.astype(np.int32, copy=False),
            gt_indptr=gt_indptr.astype(np.int64, copy=False),
            gt_indices=gt_indices.astype(np.int32, copy=False),
            ia=aspect_gt["ia"].astype(np.float32, copy=False),
            discounts=discounts.astype(np.float32, copy=False),
            k=k_eval,
        )

        ndcg = float(ndcg_sum) / float(max(1, int(n_eval)))
        out_rows.append((str(aspect), float(ndcg), int(n_eval)))

    if not out_rows:
        return pd.DataFrame(columns=["aspect", "ndcg", "n_eval"])

    return pd.DataFrame(out_rows, columns=["aspect", "ndcg", "n_eval"])
