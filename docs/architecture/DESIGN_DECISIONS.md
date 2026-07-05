# Design decisions

## Accepted inherent complexity (radon C+ register)

The project targets radon grade A/B. A few functions sit at C+ for reasons inherent to the
problem, not organisational, so decomposing them would hurt readability or risk a measured
optimisation. They are accepted and documented here; check this register before flagging a
C+ method as tech debt.

| Function | Grade | Why it is inherent |
|---|---|---|
| `block/_compiled.py::_var_residual_reduce_kernel` | C | Fused recursive-VAR njit kernel: the VAR recurrence, burn-in gating, and an explicit triple-loop matvec chosen over `@` to avoid a per-step temporary (a measured allocation win). Inline helpers add no readability in an njit kernel. |
| `block/_compiled.py::_ar_residual_reduce_kernel` | B | Univariate analogue of the VAR kernel; the same inherent recurrence and gating. |
| `api.py::_coerce_panel` | C | Nested-data validation across the two accepted ragged-panel input forms; the branching mirrors the data shape. |
| `uq/adaptive.py::_validate_agaci_inputs` | C | A flat block of independent precondition guards, each a trivial raise (the fault-isolation pattern); extracting it is what dropped `agaci_bounds` itself to grade A. |

Deliberately NOT accepted here: `bootstrap` / `bootstrap_reduce` and the `backend == "compiled"`
isinstance dispatch, whose complexity is slated to dissolve when the executor dispatch is
unified into a registry, so they are not accepted permanently.

## Compiled RNG stream stability

The compiled backend uses an opt-in counter-based Philox stream that is equal in distribution
to the default PCG64 stream but not bit-identical to it. Its exact byte stream is pinned by a
known-answer test on the round function and an internal regression golden on the index stream,
but is NOT guaranteed stable across versions (it may change, for example to align the integer
index stream with a JAX philox4x32 backend). The default numpy backend stays byte-frozen. The
authoritative statement of this contract is the `tsbootstrap.block._compiled` module docstring.
