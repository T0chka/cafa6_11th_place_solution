"""Small diagnostics used by postprocessing."""

import numpy as np


def print_state_stats(state, step_name: str) -> None:
    n_proteins = int(state.indptr.size - 1)
    terms_per_protein = state.indptr[1:] - state.indptr[:-1]
    if n_proteins == 0:
        print(f"[DEBUG] {step_name:<30}: N prots=0")
        return
    print(
        f"[DEBUG] {step_name:<30}: N prots={n_proteins} | "
        f"Terms/prot: min={int(terms_per_protein.min())}, "
        f"med={float(np.median(terms_per_protein)):.1f}, "
        f"max={int(terms_per_protein.max())}"
    )
