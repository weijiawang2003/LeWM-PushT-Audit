"""Synthetic trace producer for exercising the extracted numerical auditor.

The sampler/update bookkeeping follows archived complete_query. Neural terminal
embeddings are replaced by a declared deterministic array mapping. The output
is a software fixture, with no robot task, model weights, or performance result.
"""

import numpy as np

from .extracted import norm, unnorm


def seeded_rngs():
    return [np.random.default_rng(34199 + seed) for seed in range(88000, 88008)]


def pack_raw_commands(raw_u):
    """Normalize five chronological 2D commands, then flatten to a 10D block."""
    raw_u = np.asarray(raw_u, dtype=np.float32)
    if raw_u.shape[-2:] != (5, 2):
        raise ValueError("raw command blocks must end in shape (5, 2)")
    return norm(np, raw_u).reshape(*raw_u.shape[:-2], 10)


def pack_repeated_knots(knots_u):
    """Use the controller's distinct recipe: repeat each knot five times."""
    knots_u = np.asarray(knots_u, dtype=np.float32)
    if knots_u.shape[-1] != 2:
        raise ValueError("knots must end in shape (2,)")
    return np.repeat(norm(np, knots_u)[..., None, :], 5, axis=-2).reshape(
        *knots_u.shape[:-1], 10
    )


def synthetic_trace(previous_selected=None, *, tie_costs=False, rngs=None):
    """Record the original B8, C64, three-round arithmetic on synthetic arrays."""
    warm = (
        np.zeros((8, 3, 2), dtype=np.float32)
        if previous_selected is None
        else np.concatenate((previous_selected[:, 1:], previous_selected[:, -1:]), axis=1)
    )
    mean = np.zeros_like(warm) if previous_selected is None else norm(np, warm)
    std = np.ones_like(mean)
    goals = np.random.default_rng(781).normal(0, 0.3, (8, 192)).astype(np.float32)
    data = {
        "goal_E": goals,
        "initial_mean_z": mean.copy(),
        "warm_candidate_u": warm.copy(),
    }
    best_cost = np.full(8, np.inf, dtype=np.float32)
    best_id = np.full(8, -1, dtype=np.int64)
    best_u = np.zeros_like(mean)
    rngs = seeded_rngs() if rngs is None else rngs
    for iteration in range(3):
        noise = np.stack([rng.standard_normal((64, 3, 2)) for rng in rngs])
        z = (mean[:, None] + std[:, None] * noise).astype(np.float32)
        u = np.clip(unnorm(np, z), -1, 1)
        u[:, 0] = 0.
        u[:, 1] = warm
        actual_z = norm(np, u)
        # Deliberate mock embedding: repeat six action coordinates 32 times.
        # This is not a learned forward pass or an environmental prediction.
        terminal = np.tile(u.reshape(8, 64, 6), (1, 1, 32))
        if tie_costs:
            terminal.fill(0.)
        cost = ((terminal[:, :, None] - goals[None, None]) ** 2).mean(-1).min(-1)
        elite = np.argsort(cost, axis=1, kind="stable")[:, :8]
        prefix = f"i{iteration}_"
        data.update({
            prefix + "noise_f64": noise,
            prefix + "actual_z": actual_z,
            prefix + "actual_u": u,
            prefix + "terminal_E": terminal,
            prefix + "cost": cost,
            prefix + "elite": elite,
            prefix + "mean_before": mean.copy(),
            prefix + "std_before": std.copy(),
        })
        for slot in range(8):
            chosen = int(np.argmin(cost[slot]))
            if cost[slot, chosen] < best_cost[slot]:
                best_cost[slot] = cost[slot, chosen]
                best_id[slot] = iteration * 64 + chosen
                best_u[slot] = u[slot, chosen]
            values = actual_z[slot, elite[slot]]
            mean[slot] = .25 * mean[slot] + .75 * values.mean(0)
            std[slot] = np.maximum(.1, .25 * std[slot] + .75 * values.std(0))
        data.update({prefix + "mean_after": mean.copy(), prefix + "std_after": std.copy()})
    data.update({
        "selected_u": best_u,
        "selected_global_id": best_id,
        "selected_cost": best_cost,
        "selected_first5_u": np.repeat(best_u[:, 0, None, :], 5, axis=1),
        "shifted_warm_u": np.concatenate((best_u[:, 1:], best_u[:, -1:]), axis=1),
    })
    return data

