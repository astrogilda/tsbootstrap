"""Compiled recursive AR, VAR and SieveAR kernels: values, reduce, wild innovations, exog.

The same-stream oracle rebuilds a replicate's IID innovation indices from the compiled
IID index kernel (the AR/VAR kernel draws one bounded integer per generated step from
the same replicate stream, series 0) and runs the numpy simulation engines on them.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import ks_2samp

pytest.importorskip("numba")

from tsbootstrap.api import bootstrap, bootstrap_reduce  # noqa: E402
from tsbootstrap.block import _compiled as sk  # noqa: E402
from tsbootstrap.engines.arma_scipy import simulate_ar_batched  # noqa: E402
from tsbootstrap.engines.var import simulate_var_batched  # noqa: E402
from tsbootstrap.methods import (  # noqa: E402
    AR,
    VAR,
    BlockWild,
    ResidualBootstrap,
    SieveAR,
    Wild,
)
from tsbootstrap.model.recursive import _prepare_residual, _prepare_sieve  # noqa: E402


def _root(seed: int) -> tuple[int, int]:
    words = np.random.SeedSequence(seed).generate_state(4, dtype=np.uint32)
    return (int(words[0]) << 32) | int(words[1]), (int(words[2]) << 32) | int(words[3])


def _ar2(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = 0.5 * x[t - 1] - 0.2 * x[t - 2] + rng.standard_normal()
    return x + 3.0


def _var2(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.array([[0.5, 0.1], [-0.2, 0.3]])
    x = np.zeros((n, 2))
    for t in range(1, n):
        x[t] = a @ x[t - 1] + rng.standard_normal(2)
    return x


def _ar_ctx(series, innovation=None, exog=None, order=2):
    kwargs = {} if innovation is None else {"innovation": innovation}
    spec = ResidualBootstrap(model=AR(order=order), **kwargs)
    return _prepare_residual(np.ascontiguousarray(series), spec, exog)


def test_ar_values_match_the_same_stream_numpy_simulation():
    n, B, p = 150, 40, 2
    x = _ar2(n, 1)
    ctx = _ar_ctx(x)
    root = _root(5)
    vals = sk.ar_residual_values(ctx, root, n, n_bootstraps=B)
    assert vals.shape == (B, n, 1)
    idx = sk.iid_indices(n - p, root, n_bootstraps=B)
    e_star = ctx.resampling_innovations[idx]
    inits = np.tile(x[:p], (B, 1))
    ref = simulate_ar_batched(ctx.fit.ar_coefs, ctx.fit.intercept, inits, e_star)[:, :n]
    np.testing.assert_allclose(vals[:, :, 0], ref, rtol=1e-12, atol=1e-12)


def test_ar_values_with_exog_match_the_same_stream_numpy_simulation():
    n, B, p = 120, 20, 2
    x = _ar2(n, 2)
    z = np.random.default_rng(3).standard_normal((n, 2))
    ctx = _ar_ctx(x, exog=z)
    root = _root(6)
    vals = sk.ar_residual_values(ctx, root, n, n_bootstraps=B)
    idx = sk.iid_indices(n - p, root, n_bootstraps=B)
    e_star = ctx.resampling_innovations[idx] + (z[p:] @ ctx.exog_state.coefs)[None]
    ref = simulate_ar_batched(ctx.fit.ar_coefs, ctx.fit.intercept, np.tile(x[:p], (B, 1)), e_star)
    np.testing.assert_allclose(vals[:, :, 0], ref[:, :n], rtol=1e-12, atol=1e-12)


def test_var_values_match_the_same_stream_numpy_simulation():
    n, B, p = 160, 30, 1
    x = _var2(n, 4)
    spec = ResidualBootstrap(model=VAR(order=p))
    ctx = _prepare_residual(np.ascontiguousarray(x), spec, None)
    root = _root(7)
    vals = sk.var_residual_values(ctx, root, n, n_bootstraps=B)
    assert vals.shape == (B, n, 2)
    idx = sk.iid_indices(n - p, root, n_bootstraps=B)
    e_star = ctx.resampling_innovations[idx]
    inits = np.tile(x[:p], (B, 1, 1))
    ref = simulate_var_batched(ctx.fit.coefs, ctx.fit.intercept, inits, e_star)[:, :n]
    np.testing.assert_allclose(vals, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("reducer", ["mean", "var", "std", "quantile"])
def test_reduce_equals_the_reducer_applied_to_values(reducer):
    n, B = 120, 50
    ctx = _ar_ctx(_ar2(n, 8), innovation=Wild(distribution="gaussian"))
    root = _root(9)
    vals = sk.ar_residual_values(ctx, root, n, n_bootstraps=B)[:, :, 0]
    q = 0.3 if reducer == "quantile" else None
    red = sk.ar_residual_reduce(ctx, root, n, reducer=reducer, q=q, n_bootstraps=B)[:, 0]
    ref = {
        "mean": lambda v: v.mean(axis=1),
        "var": lambda v: v.var(axis=1),
        "std": lambda v: v.std(axis=1),
        "quantile": lambda v: np.quantile(v, 0.3, axis=1),
    }[reducer](vals)
    np.testing.assert_allclose(red, ref, rtol=1e-10, atol=1e-12)


def test_float32_values_are_the_rounded_float64_values():
    n, B = 90, 16
    ctx = _ar_ctx(_ar2(n, 10))
    root = _root(11)
    v64 = sk.ar_residual_values(ctx, root, n, n_bootstraps=B)
    v32 = sk.ar_residual_values(ctx, root, n, np.dtype(np.float32), n_bootstraps=B)
    assert v32.dtype == np.float32
    np.testing.assert_array_equal(v32, v64.astype(np.float32))


def _multipliers(ctx, vals, n):
    """Recover m_s = e*_s / eps[s] from an AR path (fixed init, burn_in 0, no exog)."""
    p = ctx.fit.order
    path = vals[:, :, 0]
    pred = ctx.fit.intercept + sum(
        ctx.fit.ar_coefs[j] * path[:, p - 1 - j : n - 1 - j] for j in range(p)
    )
    return (path[:, p:] - pred) / ctx.resampling_innovations[None, :]


@pytest.mark.parametrize("dist", ["rademacher", "gaussian", "mammen"])
def test_wild_multipliers_have_the_declared_law(dist):
    n, B = 200, 400
    ctx = _ar_ctx(_ar2(n, 12), innovation=Wild(distribution=dist))
    vals = sk.ar_residual_values(ctx, _root(13), n, n_bootstraps=B)
    m = _multipliers(ctx, vals, n).ravel()
    assert abs(m.mean()) < 0.02
    assert abs(m.var() - 1.0) < 0.03
    if dist == "rademacher":
        np.testing.assert_allclose(np.abs(m), 1.0, atol=1e-6)
    elif dist == "mammen":
        lo, hi = -(np.sqrt(5.0) - 1.0) / 2.0, (np.sqrt(5.0) + 1.0) / 2.0
        near = np.minimum(np.abs(m - lo), np.abs(m - hi))
        assert near.max() < 1e-6
        assert abs((m**3).mean() - 1.0) < 0.05
    else:
        kurt = (m**4).mean() / m.var() ** 2
        assert abs(kurt - 3.0) < 0.1


def test_block_wild_holds_each_multiplier_over_its_block():
    n, B, L = 200, 50, 7
    ctx = _ar_ctx(_ar2(n, 14), innovation=BlockWild(block_length=L))
    vals = sk.ar_residual_values(ctx, _root(15), n, n_bootstraps=B)
    m = _multipliers(ctx, vals, n)
    for start in range(0, m.shape[1], L):
        blk = m[:, start : start + L]
        np.testing.assert_allclose(blk, np.broadcast_to(blk[:, :1], blk.shape), atol=1e-6)


def test_var_wild_uses_one_multiplier_per_step_for_every_component():
    n, B = 150, 20
    x = _var2(n, 16)
    spec = ResidualBootstrap(model=VAR(order=1), innovation=Wild())
    ctx = _prepare_residual(np.ascontiguousarray(x), spec, None)
    vals = sk.var_residual_values(ctx, _root(17), n, n_bootstraps=B)
    pred = ctx.fit.intercept + np.einsum("ik,btk->bti", ctx.fit.coefs[0], vals[:, :-1])
    m = (vals[:, 1:] - pred) / ctx.resampling_innovations[None]
    np.testing.assert_allclose(m[..., 0], m[..., 1], atol=1e-6)
    np.testing.assert_allclose(np.abs(m), 1.0, atol=1e-6)


def test_sieve_values_and_reduce_through_the_public_api():
    x = _ar2(300, 18)
    res = bootstrap(x, method=SieveAR(), n_bootstraps=64, random_state=0, backend="compiled")
    assert res.values().shape == (64, 300)
    assert res.indices() is None
    red = bootstrap_reduce(
        x, method=SieveAR(), statistic="mean", n_bootstraps=64, random_state=0, backend="compiled"
    )
    assert red.statistics is not None
    np.testing.assert_allclose(red.statistics, res.values().mean(axis=1), rtol=1e-10)
    ctx = _prepare_sieve(np.ascontiguousarray(x), SieveAR(), None)
    assert res.values()[0, : ctx.fit.order].tolist() == x[: ctx.fit.order].tolist()


@pytest.mark.parametrize(
    "spec",
    [
        ResidualBootstrap(model=AR(order=2), innovation=Wild(distribution="mammen")),
        ResidualBootstrap(model=AR(order=2), innovation=BlockWild(block_length=5)),
        SieveAR(innovation=Wild()),
    ],
)
def test_compiled_wild_mean_statistic_matches_numpy_in_distribution(spec):
    x = _ar2(400, 19)
    kw = {"statistic": "mean", "n_bootstraps": 4000}
    ker = bootstrap_reduce(x, method=spec, random_state=1, backend="compiled", **kw).statistics
    ref = bootstrap_reduce(x, method=spec, random_state=2, **kw).statistics
    assert ker is not None and ref is not None
    assert ks_2samp(np.ravel(ker), np.ravel(ref)).pvalue > 0.001
