"""Run a small, offline synthetic arithmetic demonstration."""

import json

import numpy as np

from .extracted import audit_cem_arrays, norm, unnorm
from .fixture import pack_raw_commands, seeded_rngs, synthetic_trace


def main():
    raw = np.linspace(-0.8, 0.8, 30, dtype=np.float32).reshape(3, 5, 2)
    assert np.allclose(unnorm(np, norm(np, raw)), raw, rtol=1e-6, atol=1e-7)
    blocks = pack_raw_commands(raw)
    assert blocks.shape == (3, 10)
    trace = synthetic_trace()
    selected = audit_cem_arrays(np, trace, seeded_rngs())
    damaged = {name: value.copy() for name, value in trace.items()}
    damaged["selected_global_id"][0] += 1
    try:
        audit_cem_arrays(np, damaged, seeded_rngs())
    except AssertionError:
        corruption_rejected = True
    else:
        raise RuntimeError("synthetic selection corruption went undetected")
    print(json.dumps({
        "fixture_kind": "synthetic arithmetic only",
        "normalization_roundtrip_passed": True,
        "chronological_5x2_to_10D_packing_passed": True,
        "cem_trace_audit_passed": True,
        "candidate_plans_checked": 1536,
        "selected_shape": list(selected.shape),
        "tampered_selection_rejected": corruption_rejected,
        "neural_forward_calls": 0,
        "environment_controls": 0,
        "historical_result_reproduced": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

