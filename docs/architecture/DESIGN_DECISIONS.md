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

The executor dispatch is now unified into the `(spec type, backend)` registry (see the next
section), so `bootstrap` / `bootstrap_reduce` are thin registry lookups at grade B and the
`backend == "compiled"` isinstance dispatch no longer exists. The only C+ functions left in
`api.py` are the ragged-panel pair (`_coerce_panel`, above, and `bootstrap_reduce_panel`),
whose branching mirrors the two accepted panel input forms.

## Compiled module size (accepted, not split)

`block/_compiled.py` is a single ~2750-line module of ~31 njit kernels. It is deliberately not
split into key-primitives / kernels / dispatch sub-modules, for two reasons. First, its
maintainability index is radon grade B, which is within the accepted A/B target, so there is no
complexity trigger to act on. Second, the njit kernels are compiled with `cache=True` and call each
other in-module; numba resolves those cross-kernel calls and inlining within one compilation unit,
and splitting kernels across modules risks a compilation-boundary regression (lost inlining, extra
dispatch, cache churn) for no readability gain. The one part that is genuinely reusable and
numba-free, the counter-based key derivation, was extracted to the `prng_keys.py` reference module;
the kernels stay put by design. Revisit only if the MI degrades past grade B or a second backend
needs to share kernel code.

## Compiled RNG stream stability

The compiled backend uses an opt-in counter-based Philox stream that is equal in distribution
to the default PCG64 stream but not bit-identical to it. Its exact byte stream is pinned by a
known-answer test on the round function and an internal regression golden on the index stream,
but is NOT guaranteed stable across versions (it may change, for example to align the integer
index stream with a JAX philox4x32 backend). The default numpy backend stays byte-frozen. The
authoritative statement of this contract is the `tsbootstrap.block._compiled` module docstring.

## Executor dispatch seam

Execution is dispatched through two registries keyed by `(spec type, backend)`:
`_VALUES_EXECUTORS` (materialise the full `(B, n[, d])` sample) and `_REDUCE_EXECUTORS` (fuse
generation and reduction to `(B, |theta|)` without materialising the sample). The numpy backend
registers one per-spec chunk kernel via `register_chunk_executor`, which is wrapped into both a
values executor and the generic streaming reduce executor over the fixed `_CHUNK_SIZE` chunk
loop; the compiled backend registers its fused kernels directly under the `"compiled"` key. Each
executor owns its RNG derivation from the run's root `SeedSequence`: the numpy path spawns the
`B` children upfront, the compiled path packs the root into two words and derives per-replicate
keys in-kernel. The entry points therefore carry no backend branch, and a new backend (a JAX
threefry path, say) registers a `(spec, "jax")` pair with no change to `api.py`.

The numpy generation path (upfront child spawn plus the fixed chunk size) is byte-frozen and
unchanged by this seam; a chunk-boundary identity test pins that the chunk size is never part of
the reproducibility contract (replicate `i` binds to child `i` of the root wherever the chunk
boundary falls). The ragged-panel reduce (`bootstrap_reduce_panel`) stays off this registry on
purpose: its input contract (a flat array plus CSR offsets) and its numpy reference (a per-series
loop) differ fundamentally from the rectangular seam, so it dispatches to the compiled panel
kernel directly rather than through a shared executor.
