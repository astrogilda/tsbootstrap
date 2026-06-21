# Migrating to tsbootstrap 0.2.0

0.2.0 replaces the class-based API with a single typed entry point. This is a
breaking change: the previous bootstrap classes are removed, not aliased.

## The new entry point

```python
from tsbootstrap import bootstrap, MovingBlock

result = bootstrap(x, method=MovingBlock(block_length="auto"), n_bootstraps=999, random_state=0)
samples = result.values()          # (n_bootstraps, n[, d])
oob = result.get_oob_mask()        # (n_bootstraps, n) for observation-resampling methods
```

`bootstrap` takes the data and a method specification, and returns a
`BootstrapResult` (a sequence of `BootstrapSample` with run metadata).

## Method mapping

| Previous class | New specification |
|---|---|
| `MovingBlockBootstrap` | `MovingBlock(block_length=...)` |
| `CircularBlockBootstrap` | `CircularBlock(block_length=...)` |
| `StationaryBlockBootstrap` | `StationaryBlock(avg_block_length=...)` |
| `NonOverlappingBlockBootstrap` | `NonOverlappingBlock(block_length=...)` |
| `Bartletts/Blackman/Hamming/Hanning/Tukey` | `TaperedBlock(window=..., block_length=...)` |
| whole/block residual bootstrap | `ResidualBootstrap(model=AR(order=p))` |
| whole/block sieve bootstrap | `SieveAR()` |

Block length defaults to `"auto"` (Politis-White automatic selection). The
residual and sieve bootstraps now regenerate series recursively from the fitted
dynamics with resampled, centered innovations.

## Removed

The async execution layer, the batch public layer, the string-keyed factory, the
in-process feature-flag/rollout system, the dependency-injection container, the
statistic-preserving method, and the duplicated sklearn model wrapper are
removed. The internal service/backend scaffolding they required is gone with
them.

## Deferred to a later release

Markov-resampling and distribution bootstraps, GARCH/volatility models, ARIMA
and VAR recursive simulation, exogenous-input handling, the Narwhals DataFrame
boundary, and the sktime/skbase adapter classes are being reintroduced on the
new API. Until an sktime adapter lands, the package exposes the functional
`bootstrap()` API only.
