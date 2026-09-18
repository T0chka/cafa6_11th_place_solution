from dataclasses import dataclass
from pathlib import Path

import numba as nb
import numpy as np

from src.core.csr import csr_from_dense
from src.core.postprocess import CSRState


def load_npz_as_state(path: str | Path) -> tuple[np.ndarray, CSRState]:
    data = np.load(path, allow_pickle=True)
    indptr, indices, scores = csr_from_dense(data["term_pos"], data["scores"])
    return data["entry_ids"], CSRState(indptr=indptr, indices=indices, scores=scores)


def term_ia(aspect_gt: dict) -> np.ndarray:
    return aspect_gt["ia"][:, None]


def term_depth(aspect_gt: dict) -> np.ndarray:
    graph = aspect_gt["graph"]
    parents_indptr = graph["parents_indptr"]
    children_indptr = graph["children_indptr"]
    children_indices = graph["children_indices"]
    n_terms = int(parents_indptr.shape[0] - 1)
    indeg = (parents_indptr[1:] - parents_indptr[:-1]).astype(np.int32, copy=False)
    depth = np.full(n_terms, np.iinfo(np.int32).max, dtype=np.int32)
    queue = list(np.flatnonzero(indeg == 0).astype(np.int32, copy=False))
    for node in queue:
        depth[int(node)] = 0
    head = 0
    while head < len(queue):
        node = int(queue[head])
        head += 1
        next_depth = int(depth[node]) + 1
        for child in children_indices[int(children_indptr[node]):int(children_indptr[node + 1])]:
            child = int(child)
            if next_depth < int(depth[child]):
                depth[child] = next_depth
            indeg[child] -= 1
            if int(indeg[child]) == 0:
                queue.append(np.int32(child))
    return depth.astype(np.float32, copy=False)[:, None]


def build_extra_by_term(aspect_gt: dict, names: tuple[str, ...]) -> np.ndarray | None:
    if not names:
        return None
    builders = {"term_ia": term_ia, "term_depth": term_depth}
    return np.hstack([builders[name](aspect_gt) for name in names]).astype(np.float32, copy=False)


@dataclass(frozen=True)
class NonexpFeatures:
    axis_to_nonexp_protpos: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    bit_positions: np.ndarray
    feature_names: tuple[str, ...]


def prepare_nonexp_features(aspect_gt: dict, axis_entry_ids: np.ndarray, exclude_codes: tuple[str, ...] = ()) -> NonexpFeatures | None:
    nonexp = aspect_gt.get("nonexp_codes")
    if nonexp is None or int(nonexp["nonexp_ids"].size) == 0:
        return None
    code_names = nonexp["nonexp_code_names"].astype(object, copy=False)
    exclude = set(map(str, exclude_codes))
    keep = np.fromiter((str(code) not in exclude for code in code_names), dtype=bool, count=int(code_names.size))
    bit_positions = np.flatnonzero(keep).astype(np.uint16, copy=False)
    feature_names = tuple(f"nonexp_{str(code)}" for code in code_names[keep].tolist())
    pos_by_id = {str(entry_id): i for i, entry_id in enumerate(nonexp["nonexp_ids"].tolist())}
    axis_to_nonexp = np.fromiter((pos_by_id.get(str(entry_id), -1) for entry_id in axis_entry_ids), dtype=np.int32, count=int(axis_entry_ids.size))
    return NonexpFeatures(axis_to_nonexp, nonexp["nonexp_indptr"], nonexp["nonexp_indices"], nonexp["nonexp_data"], bit_positions, feature_names)


def gather_nonexp_bitmask(prots_idx: np.ndarray, term_indptr: np.ndarray, term_indices: np.ndarray, nonexp: NonexpFeatures) -> np.ndarray:
    out = np.zeros(int(term_indices.size), dtype=np.uint16)
    for q in range(int(prots_idx.size)):
        nonexp_pos = int(nonexp.axis_to_nonexp_protpos[int(prots_idx[q])])
        if nonexp_pos < 0:
            continue
        i = int(term_indptr[q])
        e = int(term_indptr[q + 1])
        j = int(nonexp.indptr[nonexp_pos])
        ne = int(nonexp.indptr[nonexp_pos + 1])
        while i < e and j < ne:
            t = int(term_indices[i])
            u = int(nonexp.indices[j])
            if t == u:
                out[i] = np.uint16(nonexp.data[j])
                i += 1
                j += 1
            elif t < u:
                i += 1
            else:
                j += 1
    return out


def expand_bitmask_features(mask: np.ndarray, bit_positions: np.ndarray) -> np.ndarray:
    if int(bit_positions.size) == 0:
        return np.empty((int(mask.size), 0), dtype=np.float32)
    shifted = np.asarray(mask, dtype=np.uint16)[:, None] >> np.asarray(bit_positions, dtype=np.uint16)[None, :]
    return (shifted & np.uint16(1)).astype(np.float32, copy=False)


@nb.njit
def labels_from_gt(prots_idx: np.ndarray, term_indptr: np.ndarray, term_indices: np.ndarray, gt_indptr: np.ndarray, gt_indices: np.ndarray) -> np.ndarray:
    out = np.empty(int(term_indices.size), dtype=np.float32)
    off = 0
    for q in range(int(prots_idx.size)):
        prot = int(prots_idx[q])
        gs = int(gt_indptr[prot])
        ge = int(gt_indptr[prot + 1])
        g = gs
        for k in range(int(term_indptr[q]), int(term_indptr[q + 1])):
            term = int(term_indices[k])
            while g < ge and int(gt_indices[g]) < term:
                g += 1
            out[off] = np.float32(1.0 if g < ge and int(gt_indices[g]) == term else 0.0)
            off += 1
    return out


def drop_queries_without_signal(groups: np.ndarray, x: np.ndarray, prots_idx: np.ndarray, term_indptr: np.ndarray, term_indices: np.ndarray, y: np.ndarray, n_members: int):
    keep = np.zeros(int(groups.size), dtype=bool)
    for q in range(int(groups.size)):
        s = int(term_indptr[q])
        e = int(term_indptr[q + 1])
        if s == e:
            continue
        if float(y[s:e].min()) == float(y[s:e].max()):
            continue
        if float(x[s:e, :n_members].min()) == float(x[s:e, :n_members].max()):
            continue
        keep[q] = True
    new_groups = groups[keep]
    new_prots = prots_idx[keep]
    counts = (term_indptr[1:] - term_indptr[:-1])[keep]
    new_indptr = np.empty(int(new_groups.size) + 1, dtype=np.int64)
    new_indptr[0] = 0
    np.cumsum(counts, out=new_indptr[1:])
    n = int(new_indptr[-1])
    new_terms = np.empty(n, dtype=term_indices.dtype)
    new_y = np.empty(n, dtype=y.dtype)
    new_x = np.empty((n, x.shape[1]), dtype=x.dtype)
    off = 0
    for q in range(int(groups.size)):
        if not keep[q]:
            continue
        s = int(term_indptr[q])
        e = int(term_indptr[q + 1])
        m = e - s
        new_terms[off:off + m] = term_indices[s:e]
        new_y[off:off + m] = y[s:e]
        new_x[off:off + m] = x[s:e]
        off += m
    return new_groups, new_x, new_prots, new_indptr, new_terms, new_y
