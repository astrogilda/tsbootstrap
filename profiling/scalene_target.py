"""Entry point for scalene and py-spy (they wrap a script, not a function).

scalene splits CPU time into Python vs native (numpy/BLAS) and tracks memory
line-by-line across the real ``src`` functions, the best single view of "are we
stuck in slow Python where native would do".

    scalene --html --outfile profiling/REPORTS/scalene.html -m profiling.scalene_target
    py-spy record -o profiling/REPORTS/flame.svg -- python -m profiling.scalene_target
"""

from profiling.workloads import run_all_workloads

if __name__ == "__main__":
    run_all_workloads()
