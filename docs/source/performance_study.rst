Performance study
=================

This page records what was measured while the allocation, result, random-stream
and compiled-model paths were rebuilt, and what each measurement decided. Every
number comes from a committed harness (``benchmarks/bench_perf_study.py``,
``benchmarks/study_sieve_order_cap.py``, ``benchmarks/bench_vs_arch.py``) run on
a Hetzner cloud box with 8 vCPU (AMD EPYC-Rome) with the box load average
recorded in each result's provenance block. Times are medians of three fresh
processes unless a row says otherwise.

.. contents::
   :local:
   :depth: 2


A process pool does not pay
---------------------------

The design allowed a standard-library process pool "only for workloads still
slow after the single-process fixes". It was built as a prototype, driven
through the same replicate streams as the single-process code, and measured on
the three model methods at n=2,000 with four workers:

.. list-table:: Reduce of the mean, wall seconds
   :header-rows: 1

   * - Method
     - Process pool, 4 workers
     - Single process, NumPy
     - Single process, compiled
   * - AR, B=20,000
     - 1.138
     - 0.904
     - 0.108
   * - VAR, B=10,000
     - 1.411
     - 1.098
     - 0.109
   * - SieveAR, B=20,000
     - 1.104
     - 0.854
     - 0.077

The pool loses to the single-process NumPy path by 26 to 29 percent on every
method, because start-up, task pickling and the copy back cost more than the
chunks save, and it loses to the compiled path by a factor of 10 to 14. The box
was not quiet for this run (one-minute load average 7.6 at the start on 8
cores), which weakens the absolute times but not the ordering: the pool is slower
on a loaded box because it competes for the same cores as the work it distributes.

The pool is therefore closed as measured-loses and is not shipped. Two facts
stay on record for anyone who reopens it. A pool on mixed hardware inherits the
numerical-engine difference between machines (the byte-identity guarantee between
the NumPy and compiled engines is per machine) until the cross-machine work item
lands. And the compiled backend already uses every core through numba threads, so
a pool would have to beat threads, not a single core.


Memory peak of a run
--------------------

Chunks are now written straight into the one output buffer and the replicate
streams are drawn in row groups, so a run never holds more than one chunk budget
of temporaries beyond the output itself. The test ``tests/unit/test_memory_peak.py``
asserts that the extra peak, in units of the chunk budget, stays at or below 1.0:

.. list-table:: Peak above the held output, in chunk-budget units (assert at most 1.0)
   :header-rows: 1

   * - Method
     - Extra peak
   * - IID
     - 0.60
   * - MovingBlock
     - 0.83
   * - CircularBlock
     - 0.84
   * - NonOverlappingBlock
     - 0.83
   * - StationaryBlock
     - 0.60
   * - TaperedBlock
     - 0.83
   * - ResidualBootstrap (AR)
     - 0.65
   * - SieveAR
     - 0.65


Amendment: the compiled path does not store indices
---------------------------------------------------

The converged design stored the ``(B, n)`` index matrix in every
:class:`~tsbootstrap.results.BootstrapResult`. For the compiled values path that
is a second array as large as the values, written only so that a later call to
``indices`` can read it. The compiled kernels now write values in the run dtype
and keep no index matrix. ``BootstrapResult`` takes a function that regenerates
the indices on demand from the same counter-based stream, and caches the result
after the first call. The regenerated indices are identical to the ones the
kernel used, which ``tests/unit/test_compiled.py::TestLazyIndices`` checks for
every compiled values method. A recursive (model-based) compiled run has
no indices at all, and a test pins that too. The NumPy path still stores its
indices.


Latency, not only throughput
----------------------------

.. list-table:: First-call and small-call costs (one run, quiet box)
   :header-rows: 1

   * - Measurement
     - Seconds
   * - ``import tsbootstrap``, numba importable
     - 1.34
   * - ``import tsbootstrap``, numba blocked
     - 1.08
   * - First ``bootstrap`` call, IID, NumPy backend
     - 0.0047
   * - First compiled call, IID, empty numba cache
     - 3.45
   * - First compiled call, IID, warm numba cache
     - 0.23
   * - First compiled call, VAR, empty numba cache
     - 11.1
   * - One ``bootstrap`` call at B=1, IID
     - 0.00098
   * - One ``bootstrap`` call at B=1, MovingBlock
     - 0.00114
   * - One ``bootstrap`` call at B=1, AR
     - 0.00205
   * - Named reducer, callable reducer, vectorised callable (B=999 chunk set)
     - 0.068, 0.069, 0.049
   * - Panel of 200 series, per series
     - 0.0051
   * - Added cost of one chunk
     - 0.0012

The kernel warm-up was deleted: each kernel now compiles on its first use, so a
user who never calls a compiled kernel never pays for it, and the numba cache
makes the second process cheap (0.23 s against 3.45 s).

``conf_int`` on a series of 500 points with B=999 takes 0.020 s for the
percentile interval, 0.020 s for the basic interval, 0.022 s for BCa (the
jackknife alone is 0.005 s) and 0.046 s for the studentized interval with a named
statistic. Before the chunked block jackknife the studentized interval took 0.90 s,
47 times the percentile cost (see the end of the compiled-values section).


The SieveAR order cap
---------------------

``select_ar_order`` caps the largest candidate order at ``(n - 2) // 3`` so every
candidate keeps at least two effective observations per parameter. The question
was whether the corrected AIC (AICc), which charges a high order for the lost
degrees of freedom itself, makes the cap unnecessary.
``benchmarks/study_sieve_order_cap.py`` draws 2,000 series for each of 18 cells
(AR(1), AR(2) and a near-unit-root AR(1) at n from 20 to 200), selects an order
under four rules and fits the model at the selected order.

* **Dropping the cap does not help.** AICc without the cap has a larger mean
  absolute error of the innovation variance than AICc with the cap in 18 of 18
  cells, and its non-stationary share is never lower (equal or higher in 18 of
  18 cells).
* **The shipped rule is the right default.** BIC with the cap has the smallest
  variance error in 12 of 18 cells, including every cell at n of 60 or more.
  AICc with the cap wins the six cells at n of 20 and 30. At n=20 on an AR(1)
  series its non-stationary share is 0.015 against 0.071 for BIC, and it names
  the true order in 85.1 percent of draws against 70.3 percent.

The decision is to keep the cap and keep BIC as the default. AICc with the cap is
a better rule below about n=40, so offering it as a named criterion is tracked as
a separate item rather than changing the default on a study that covers three
data-generating processes.


Before and after
----------------

The same grid was run before the branch's first change (commit ``2140f5c``) and at
``59eb3ce`` on the same box type, each with a one-minute load average under 3 at
the start. Times are wall seconds for ``bootstrap(...).values()`` unless a row
says otherwise (compiled rows use the compiled backend).

.. list-table:: Materialised values, seconds
   :header-rows: 1

   * - Workload
     - Before
     - After
     - Ratio
   * - IID, n=200, B=100,000
     - 2.498
     - 0.784
     - 3.2x
   * - MovingBlock, n=200, B=100,000
     - 2.451
     - 0.757
     - 3.2x
   * - StationaryBlock, n=200, B=100,000
     - 4.734
     - 1.345
     - 3.5x
   * - AR residual, n=200, B=100,000
     - 2.804
     - 1.266
     - 2.2x
   * - SieveAR, n=200, B=100,000
     - 2.889
     - 1.234
     - 2.3x
   * - VAR(1) residual, d=3, n=200, B=20,000
     - 0.636
     - 0.336
     - 1.9x
   * - IID, n=2,000, B=10,000
     - 0.402
     - 0.199
     - 2.0x
   * - MovingBlock, n=2,000, B=10,000
     - 0.365
     - 0.176
     - 2.1x
   * - StationaryBlock, n=2,000, B=10,000
     - 0.851
     - 0.514
     - 1.7x
   * - AR residual, n=2,000, B=10,000
     - 0.574
     - 0.562
     - 1.0x
   * - SieveAR, n=20,000, B=1,000
     - 0.346
     - 0.292
     - 1.2x
   * - IID, n=200, B=100,000, compiled
     - 0.250
     - 0.043
     - 5.8x
   * - MovingBlock, n=200, B=100,000, compiled
     - 0.251
     - 0.020
     - 12x
   * - StationaryBlock, n=200, B=100,000, compiled
     - 0.271
     - 0.057
     - 4.8x

Compiled values for AR, SieveAR and VAR did not exist before (they take 0.050,
0.043 and 0.027 s on the n=200 workloads after). SieveAR order selection takes 0.008 s, 0.038 s and 0.45 s at n=2,000, 20,000 and
200,000 (0.88 s when first measured at n=200,000, against 9.3 s for a per-order
SVD fit in the same study).

One result looked like it went the wrong way, and the cause is now measured. The
compiled **reduce** for AR and VAR at n=2,000 read 0.088 and 0.093 s against 0.050
and 0.056 s before. Stream version 2 is not the cause: it was already in place at the
baseline commit, and both commits produce the identical checksum (-1404.7757842208534
for AR). The cause is machine load. The reduce kernel runs on all eight cores, so its
wall time is that of the slowest thread, and the "before" run had a load average
near 1 while the "after" run had about 3. Run alternately on one box (A B A B A B,
three rounds, load 0.2 at the start and 5 to 6 later from other jobs), the two
commits give the same wall time on the quiet first round (AR 0.075 s on both; VAR
0.065 s on both), and wall times on the loaded rounds spread from 0.07 to 0.105 s for
either commit. The work itself, measured as CPU time summed over the threads, differs
by a small amount: AR 0.37 to 0.39 s before and 0.40 to 0.41 s after (about 5 to 8
percent more), VAR 0.33 to 0.35 s before and 0.34 to 0.36 s after (about 3 percent
more). Single-threaded the AR gap is 4 to 6 percent and appears at the commit that
added the AR values kernel, which generalised the recurrence (a full path array,
a forcing term and a wild-multiplier branch in place of a rolling window). The
benchmark now records CPU seconds beside wall seconds for the compiled cells, so a
comparison across runs with different load is no longer read from wall time alone.
The NumPy reduce for AR and SieveAR got faster (1.12 to 0.90 s and 1.12 to 0.88 s)
and the NumPy VAR reduce is unchanged (1.19 and 1.20 s).

The studentized interval had a different cause, also measured. ``conf_int`` called the
block jackknife once per replicate, and each call deleted every block with ``np.delete``
and ran the reducer on it: about 50 reducer calls per replicate, 18 microseconds
each, 999 times. The chunked path now deletes each block for a whole chunk of
replicates and calls the reducer once per block, so the Python-call count falls from
B x g to g per chunk. For a named statistic (``"mean"``, ``"var"``, ``"std"`` or a
quantile tuple) the studentized interval at n=500, B=999 takes 0.046 s instead of
0.90 s (the percentile interval is 0.020 s, so the ratio falls from 47 to 2.3). A
statistic passed as a Python callable keeps the per-replicate path, because it cannot
be assumed to accept a chunk.


Speed against ``arch``
----------------------

The compiled named-statistic reduce was re-measured against ``arch.apply`` on the
same box in two separate runs of the 16-cell grid. The published table is the
lower speedup of the two runs in every cell, and the README table and figure are
generated from those two receipts. Across the eight n=2,000 cells the compiled
reduce is 7.0 to 37 times faster than ``arch``. The two runs differ by up to 1.8
times in a single cell on a shared-core box, which is why the lower value is the
one published. The full grid and the preflight conditions of each run are in
`the benchmark guide
<https://github.com/astrogilda/tsbootstrap/blob/main/benchmarks/README.md>`_.
