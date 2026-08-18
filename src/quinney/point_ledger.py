"""Per-material-point scalar bookkeeping, keyed by id.

Two quantities need carrying between steps that Kratos will not carry itself:
accumulated damage (MPM has no damage variable at all) and the previous
equivalent plastic strain (MP_DELTA_PLASTIC_STRAIN is registered but not
implemented, so the increment has to be differenced here). The point set shrinks whenever separated
points are erased, which makes array position meaningless between steps. Every
lookup and write therefore goes through the point's id.

Pure functions over a plain mapping — no Kratos import, so replay can rebuild
a damage field on a machine with no solver installed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

PointLedger = Mapping[int, float]


def values_for(ledger: PointLedger, ids: Sequence[int]) -> NDArray[np.float64]:
    """Recorded values for the given point ids, in the order given.

    Points absent from the ledger read as zero — a newly created material
    point has no history, and zero is the right start for both damage and
    plastic strain.
    """
    return np.array([ledger.get(point_id, 0.0) for point_id in ids], dtype=np.float64)


def updated_ledger(
    ledger: PointLedger,
    ids: Sequence[int],
    values: ArrayLike,
) -> dict[int, float]:
    """Ledger with the given points' values replaced, as a new mapping.

    Points not named in ``ids`` keep their recorded value, including points
    erased in an earlier step: the record of *why* they went is worth more
    than the space it costs.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(ids) != values.size:
        raise ValueError(f"ids and values disagree in length: {len(ids)} ids, {values.size} values")

    updated = dict(ledger)
    updated.update(zip(ids, values.tolist(), strict=True))
    return updated
