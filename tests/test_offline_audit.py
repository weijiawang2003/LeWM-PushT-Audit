import numpy as np
import pytest

from wm_interface_audit.extracted import audit_cem_arrays, norm, unnorm
from wm_interface_audit.fixture import (
    pack_raw_commands,
    pack_repeated_knots,
    seeded_rngs,
    synthetic_trace,
)


def test_physical_zero_is_not_normalized_zero():
    physical_zero = np.zeros((1, 2), dtype=np.float32)
    expected = np.array([[0.0087 / 0.2019, -0.0068 / 0.2002]], dtype=np.float32)
    np.testing.assert_allclose(norm(np, physical_zero), expected, rtol=1e-6)
    assert not np.array_equal(norm(np, physical_zero), physical_zero)


def test_normalization_inverse_restores_raw_commands():
    raw = np.array([[-1., 1.], [0., 0.], [0.25, -0.5]], dtype=np.float32)
    np.testing.assert_allclose(unnorm(np, norm(np, raw)), raw, rtol=1e-6, atol=1e-7)


def test_chronological_packing_keeps_distinct_commands():
    raw = np.array([[0., 1.], [.2, .8], [.4, .6], [.6, .4], [.8, .2]], dtype=np.float32)
    packed = pack_raw_commands(raw)
    expected = ((raw - np.array([-.0087, .0068])) / np.array([.2019, .2002])).flatten()
    assert packed.shape == (10,)
    np.testing.assert_allclose(packed, expected, rtol=1e-6)
    assert not np.array_equal(packed, pack_repeated_knots(raw[0]))


def test_bad_command_block_is_rejected():
    with pytest.raises(ValueError, match="5, 2"):
        pack_raw_commands(np.zeros((4, 2)))


def test_seeded_cem_trace_passes_original_arithmetic_checks():
    trace = synthetic_trace()
    np.testing.assert_array_equal(audit_cem_arrays(np, trace, seeded_rngs()), trace["selected_u"])


def test_warm_start_uses_shifted_selected_knots():
    previous = synthetic_trace()["selected_u"]
    trace = synthetic_trace(previous)
    selected = audit_cem_arrays(np, trace, seeded_rngs(), previous_selected=previous)
    np.testing.assert_array_equal(trace["warm_candidate_u"], np.concatenate((previous[:, 1:], previous[:, -1:]), axis=1))
    np.testing.assert_array_equal(selected, trace["selected_u"])


def test_rng_state_persists_across_two_queries():
    producer_rngs = seeded_rngs()
    first = synthetic_trace(rngs=producer_rngs)
    second = synthetic_trace(first["selected_u"], rngs=producer_rngs)
    auditor_rngs = seeded_rngs()
    previous = audit_cem_arrays(np, first, auditor_rngs)
    audit_cem_arrays(np, second, auditor_rngs, previous_selected=previous)
    with pytest.raises(AssertionError):
        audit_cem_arrays(np, second, seeded_rngs(), previous_selected=previous)


@pytest.mark.parametrize("field", ["i0_noise_f64", "i1_cost", "selected_global_id"])
def test_corrupted_rng_cost_or_selection_is_rejected(field):
    trace = synthetic_trace()
    trace[field].flat[0] += 1
    with pytest.raises(AssertionError):
        audit_cem_arrays(np, trace, seeded_rngs())


def test_exact_cost_ties_keep_the_earliest_incumbent():
    trace = synthetic_trace(tie_costs=True)
    audit_cem_arrays(np, trace, seeded_rngs())
    np.testing.assert_array_equal(trace["selected_global_id"], np.zeros(8, dtype=np.int64))
    np.testing.assert_array_equal(trace["i0_elite"], np.tile(np.arange(8), (8, 1)))
