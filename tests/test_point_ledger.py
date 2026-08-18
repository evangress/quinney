"""Damage bookkeeping across material-point erasure.

Damage lives per material point and must survive the point set changing shape.
Once points are erased the element collection shrinks and re-orders, so damage
keyed by array index silently attaches to the wrong point — a bug that would
look like damage teleporting across the mesh. Keying by id is the fix, and
these tests pin it down.
"""

import numpy as np
import pytest

from quinney.point_ledger import updated_ledger, values_for


class TestValuesFor:
    def test_unseen_points_start_undamaged(self):
        assert values_for({}, ids=[7, 8, 9]) == pytest.approx([0.0, 0.0, 0.0])

    def test_returns_damage_in_the_order_of_the_ids_given(self):
        ledger = {7: 0.1, 8: 0.2, 9: 0.3}
        assert values_for(ledger, ids=[9, 7, 8]) == pytest.approx([0.3, 0.1, 0.2])

    def test_mixes_known_and_new_points(self):
        assert values_for({7: 0.5}, ids=[7, 42]) == pytest.approx([0.5, 0.0])

    def test_ignores_points_no_longer_present(self):
        # Erased points stay in the ledger but must not appear in the vector,
        # or the array length stops matching the element collection.
        result = values_for({7: 0.5, 8: 0.9}, ids=[7])
        assert result.shape == (1,)
        assert result == pytest.approx([0.5])


class TestUpdatedLedger:
    def test_records_damage_against_ids(self):
        ledger = updated_ledger({}, ids=[7, 8], values=np.array([0.25, 0.75]))
        assert ledger == {7: 0.25, 8: 0.75}

    def test_overwrites_previous_values(self):
        ledger = updated_ledger({7: 0.1}, ids=[7], values=np.array([0.4]))
        assert ledger == {7: 0.4}

    def test_leaves_untouched_points_alone(self):
        # A point erased earlier is not in `ids`; its recorded damage must
        # survive so a replay can still account for it.
        ledger = updated_ledger({7: 0.1, 99: 1.0}, ids=[7], values=np.array([0.4]))
        assert ledger == {7: 0.4, 99: 1.0}

    def test_does_not_mutate_the_ledger_it_is_given(self):
        original = {7: 0.1}
        updated_ledger(original, ids=[7], values=np.array([0.9]))
        assert original == {7: 0.1}

    def test_rejects_mismatched_lengths(self):
        # Silently zipping to the shorter of the two would drop points'
        # damage without a word.
        with pytest.raises(ValueError):
            updated_ledger({}, ids=[7, 8], values=np.array([0.5]))
