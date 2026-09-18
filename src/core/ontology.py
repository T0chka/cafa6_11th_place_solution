"""
OBO file parser, ontology graph builder, orphan pruning, and GT propagation.
"""

import numpy as np
import numba as nb

from collections.abc import Iterator
from dataclasses import dataclass

from .csr import pack_csr_from_pairs, invert_csr


@dataclass(frozen=True)
class OboSnapshot:
    edges_by_ns: dict[str, dict[str, list[str]]]
    alt_to_canon: dict[str, str]

@dataclass(frozen=True)
class OntologyGraph:
    term_ids: np.ndarray
    parents_indptr: np.ndarray
    parents_indices: np.ndarray
    children_indptr: np.ndarray
    children_indices: np.ndarray

def read_ia_tsv(path: str) -> dict[str, float]:
    """Read Information Accretion (IA) weights from TSV file."""
    information_accretion = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("term"):
                continue
            term_id, weight = line.split()[:2]
            information_accretion[term_id] = float(weight)
    return information_accretion

def _iter_obo_terms(
    obo_file: str,
) -> Iterator[tuple[str, str, list[str], list[str], bool]]:
    """Iterate OBO [Term] blocks as (id, namespace, parents, alt_ids, obsolete)."""
    current_id: str | None = None
    current_ns: str | None = None
    current_parents: list[str] = []
    current_alt_ids: list[str] = []
    current_obsolete = False
    in_term = False

    def flush() -> tuple[str, str, list[str], list[str], bool] | None:
        nonlocal current_id, current_ns, current_parents
        nonlocal current_alt_ids, current_obsolete, in_term
        if in_term and current_id and current_ns is not None:
            item = (
                current_id,
                current_ns,
                current_parents,
                current_alt_ids,
                current_obsolete,
            )
        else:
            item = None
        current_id = None
        current_ns = None
        current_parents = []
        current_alt_ids = []
        current_obsolete = False
        in_term = False
        return item

    with open(obo_file, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line == "[Term]":
                item = flush()
                if item is not None:
                    yield item
                in_term = True
                continue
            if line.startswith("[") and line.endswith("]") and line != "[Term]":
                item = flush()
                if item is not None:
                    yield item
                continue
            if not in_term:
                continue
            if line.startswith("id: "):
                current_id = line[4:].strip()
                continue
            if line.startswith("namespace: "):
                current_ns = line[11:].strip()
                continue
            if line.startswith("alt_id: "):
                current_alt_ids.append(line[8:].strip())
                continue
            if line.startswith("is_obsolete: "):
                current_obsolete = current_obsolete or (line[13:].strip() == "true")
                continue
            if line.startswith("is_a: "):
                parent_id = line[6:].split("!")[0].strip()
                current_parents.append(parent_id)
                continue
            if line.startswith("relationship: part_of "):
                parts = line.split()
                if len(parts) >= 3:
                    current_parents.append(parts[2].strip())
                continue

    item = flush()
    if item is not None:
        yield item

def parse_obo_snapshot(obo_file: str, namespaces: list[str]) -> OboSnapshot:
    """Parse OBO into non-obsolete raw edges by namespace and alt-id mapping."""
    edges_by_ns: dict[str, dict[str, list[str]]] = {ns: {} for ns in namespaces}
    alt_to_canon: dict[str, str] = {}
    alt_collisions: dict[str, tuple[str, str]] = {}

    for term_id, term_ns, parents, alt_ids, obsolete in _iter_obo_terms(obo_file):
        if term_ns not in namespaces:
            raise ValueError(f"Unexpected namespace in OBO: {term_ns}")
        if obsolete:
            continue

        edges_by_ns[term_ns][term_id] = parents
        for alt_id in alt_ids:
            prev = alt_to_canon.get(alt_id)
            if prev is None:
                alt_to_canon[alt_id] = term_id
            elif prev != term_id and alt_id not in alt_collisions:
                alt_collisions[alt_id] = (prev, term_id)

    if alt_collisions:
        items = sorted(alt_collisions.items(), key=lambda item: item[0])
        head = items[:20]
        msg = "\n".join(f"{alt_id}: {a} vs {b}" for alt_id, (a, b) in head)
        print(
            "[parse_obo_snapshot][WARN] alt_id maps to multiple canonical ids. "
            f"n={len(items)} (showing up to 20):\n{msg}"
        )

    return OboSnapshot(edges_by_ns=edges_by_ns, alt_to_canon=alt_to_canon)

def canonize_term(term_id: str, alt_to_canon: dict[str, str]) -> str:
    """Map term ID to its canonical ID if present."""
    canon = alt_to_canon.get(term_id)
    return canon if canon is not None else term_id

def build_ontology_graph(
    edges: dict[str, list[str]],
    alt_to_canon: dict[str, str],
) -> OntologyGraph:
    """Build indexed ontology graph from raw edges containing canonical GO term IDs."""
    term_set: set[str] = set()
    canon_parent_terms_by_child: dict[str, set[str]] = {}

    for child_id, parent_ids in edges.items():
        canon_child = canonize_term(child_id, alt_to_canon)
        term_set.add(canon_child)
        parent_term_set = canon_parent_terms_by_child.get(canon_child)
        if parent_term_set is None:
            parent_term_set = set()
            canon_parent_terms_by_child[canon_child] = parent_term_set
        for parent_id in parent_ids:
            canon_parent = canonize_term(parent_id, alt_to_canon)
            term_set.add(canon_parent)
            parent_term_set.add(canon_parent)

    term_list = sorted(term_set)
    term_ids = np.array(term_list, dtype=object)
    term_to_idx = {term_id: idx for idx, term_id in enumerate(term_list)}

    parent_idx_by_term: dict[str, list[int]] = {}
    for canon_child, parent_term_set in canon_parent_terms_by_child.items():
        if parent_term_set:
            parent_idx_by_term[canon_child] = sorted(
                int(term_to_idx[parent_term]) for parent_term in parent_term_set
            )

    row_list: list[int] = []
    col_list: list[int] = []
    for child_term, parent_idx_list in parent_idx_by_term.items():
        child_idx = int(term_to_idx[child_term])
        for parent_idx in parent_idx_list:
            row_list.append(child_idx)
            col_list.append(int(parent_idx))

    if not row_list:
        raise ValueError("No parent relationships found in the ontology graph.")

    parents_indptr, parents_indices = pack_csr_from_pairs(
        row_index=np.asarray(row_list, dtype=np.int64),
        col_index=np.asarray(col_list, dtype=np.int32),
        n_rows=int(len(term_list)),
    )

    children_indptr, children_indices = invert_csr(
        indptr=parents_indptr,
        indices=parents_indices,
        n_cols=int(len(term_list)),
    )

    return OntologyGraph(
        term_ids=term_ids,
        parents_indptr=parents_indptr,
        parents_indices=parents_indices,
        children_indptr=children_indptr,
        children_indices=children_indices,
    )

def prune_orphans(graph: OntologyGraph) -> OntologyGraph:
    """Remove ontology terms that are unreachable from any root."""
    n_terms = int(graph.term_ids.size)
    parent_sizes = graph.parents_indptr[1:] - graph.parents_indptr[:-1]
    root_indices = np.flatnonzero(parent_sizes == 0).astype(np.int32, copy=False)

    keep = np.zeros(n_terms, dtype=np.bool_)
    stack: list[int] = [int(idx) for idx in root_indices.tolist()]
    while stack:
        node_idx = int(stack.pop())
        if keep[node_idx]:
            continue
        keep[node_idx] = True
        start = int(graph.children_indptr[node_idx])
        end = int(graph.children_indptr[node_idx + 1])
        if end > start:
            stack.extend(graph.children_indices[start:end].tolist())

    kept_n = int(keep.sum())
    if kept_n == n_terms:
        print(f"[prune_orphans] no change: n_terms={n_terms}")
        return graph

    print(f"[prune_orphans] pruned: before={n_terms}, after={kept_n}")

    kept_old = np.flatnonzero(keep).astype(np.int32, copy=False)
    old_to_new = -np.ones(n_terms, dtype=np.int32)
    old_to_new[kept_old] = np.arange(kept_old.size, dtype=np.int32)

    term_ids = graph.term_ids[kept_old]
    term_list = [str(term_id) for term_id in term_ids.tolist()]

    parent_idx_by_term: dict[str, list[int]] = {}
    for old_idx in kept_old.tolist():
        new_idx = int(old_to_new[old_idx])
        start = int(graph.parents_indptr[old_idx])
        end = int(graph.parents_indptr[old_idx + 1])
        mapped = old_to_new[graph.parents_indices[start:end]]
        mapped = mapped[mapped >= 0]
        if int(mapped.size):
            parent_idx_by_term[term_list[new_idx]] = mapped.astype(
                np.int32, copy=False
            ).tolist()

    row_list: list[int] = []
    col_list: list[int] = []

    for old_child in kept_old.tolist():
        new_child = int(old_to_new[int(old_child)])
        start = int(graph.parents_indptr[int(old_child)])
        end = int(graph.parents_indptr[int(old_child) + 1])
        parents_old = graph.parents_indices[start:end]
        for parent_old in parents_old.tolist():
            new_parent = int(old_to_new[int(parent_old)])
            if new_parent >= 0:
                row_list.append(new_child)
                col_list.append(new_parent)

    if not row_list:
        raise ValueError("No parent relationships found in the ontology graph after pruning.")

    parents_indptr, parents_indices = pack_csr_from_pairs(
        row_index=np.asarray(row_list, dtype=np.int64),
        col_index=np.asarray(col_list, dtype=np.int32),
        n_rows=int(kept_old.size),
    )

    children_indptr, children_indices = invert_csr(
        indptr=parents_indptr,
        indices=parents_indices,
        n_cols=int(kept_old.size),
    )

    return OntologyGraph(
        term_ids=term_ids,
        parents_indptr=parents_indptr,
        parents_indices=parents_indices,
        children_indptr=children_indptr,
        children_indices=children_indices,
    )


@nb.njit(cache=True)
def closure_csr_by_parents(
    seed_indptr: np.ndarray,
    seed_indices: np.ndarray,
    parents_indptr: np.ndarray,
    parents_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = int(seed_indptr.size - 1)
    n_terms = int(parents_indptr.size - 1)

    gt_indptr = np.empty(n_rows + 1, dtype=np.int64)
    gt_indptr[0] = 0

    mark = np.zeros(n_terms, dtype=np.uint8)
    stack = np.empty(256, dtype=np.int32)
    out = np.empty(512, dtype=np.int32)
    touched = np.empty(512, dtype=np.int32)

    gt_indices = np.empty(1024, dtype=np.int32)
    nnz = 0

    for row in range(n_rows):
        stack_size = 0
        out_size = 0
        touched_size = 0

        s0 = int(seed_indptr[row])
        s1 = int(seed_indptr[row + 1])

        for j in range(s0, s1):
            t = int(seed_indices[j])
            if t < 0 or t >= n_terms or mark[t] != 0:
                continue
            mark[t] = 1

            if touched_size >= touched.size:
                tmp = np.empty(touched.size * 2, dtype=np.int32)
                tmp[:touched_size] = touched[:touched_size]
                touched = tmp
            touched[touched_size] = t
            touched_size += 1

            if stack_size >= stack.size:
                tmp = np.empty(stack.size * 2, dtype=np.int32)
                tmp[:stack_size] = stack[:stack_size]
                stack = tmp
            stack[stack_size] = t
            stack_size += 1

        while stack_size > 0:
            stack_size -= 1
            t = int(stack[stack_size])

            if out_size >= out.size:
                tmp = np.empty(out.size * 2, dtype=np.int32)
                tmp[:out_size] = out[:out_size]
                out = tmp
            out[out_size] = t
            out_size += 1

            p0 = int(parents_indptr[t])
            p1 = int(parents_indptr[t + 1])
            for p in range(p0, p1):
                parent = int(parents_indices[p])
                if parent < 0 or parent >= n_terms or mark[parent] != 0:
                    continue
                mark[parent] = 1

                if touched_size >= touched.size:
                    tmp = np.empty(touched.size * 2, dtype=np.int32)
                    tmp[:touched_size] = touched[:touched_size]
                    touched = tmp
                touched[touched_size] = parent
                touched_size += 1

                if stack_size >= stack.size:
                    tmp = np.empty(stack.size * 2, dtype=np.int32)
                    tmp[:stack_size] = stack[:stack_size]
                    stack = tmp
                stack[stack_size] = parent
                stack_size += 1

        for k in range(touched_size):
            mark[touched[k]] = 0

        out_slice = out[:out_size]
        out_slice.sort()

        need = nnz + out_size
        if need > gt_indices.size:
            new_size = gt_indices.size
            while new_size < need:
                new_size *= 2
            tmp = np.empty(new_size, dtype=np.int32)
            tmp[:nnz] = gt_indices[:nnz]
            gt_indices = tmp

        gt_indices[nnz:need] = out_slice
        nnz = need
        gt_indptr[row + 1] = nnz

    return gt_indptr, gt_indices[:nnz]
