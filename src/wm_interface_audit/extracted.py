"""Portable arithmetic extracted from the local saved-query audit.

No model inference or native control occurs here. See PROVENANCE.json for the
verbatim source functions and unchanged CEM block used by audit_cem_arrays.
Assertions deliberately preserve the archived audit; do not use Python -O.
"""

MU, SIGMA = (-.0087, .0068), (.2019, .2002)

def norm(np, u):
    return ((u - np.asarray(MU, dtype=np.float32)) / np.asarray(SIGMA, dtype=np.float32)).astype(np.float32)


def unnorm(np, z):
    return (z * np.asarray(SIGMA, dtype=np.float32) + np.asarray(MU, dtype=np.float32)).astype(np.float32)


def shape_dtype(np, value, shape, dtype):
    assert value.shape == shape and value.dtype == dtype and np.isfinite(value).all()


def audit_cem_arrays(np, data, rngs, previous_selected=None):
    """Recompute one B8/64-candidate/three-iteration saved CEM query."""
    warm = np.zeros((8, 3, 2), dtype=np.float32) if previous_selected is None else np.concatenate((previous_selected[:, 1:], previous_selected[:, -1:]), axis=1)
    mean = np.zeros_like(warm) if previous_selected is None else norm(np, warm)
    std = np.ones_like(mean)
    assert np.array_equal(data['initial_mean_z'], mean) and np.array_equal(data['warm_candidate_u'], warm)
    best_cost = np.full(8, np.inf, dtype=np.float32)
    best_id = np.full(8, -1, dtype=np.int64)
    best_u = np.zeros((8, 3, 2), dtype=np.float32)
    for iteration in range(3):
        prefix = f'i{iteration}_'
        noise = np.stack([rng.standard_normal((64, 3, 2)) for rng in rngs])
        shape_dtype(np, data[prefix + 'noise_f64'], (8, 64, 3, 2), np.float64)
        assert np.array_equal(data[prefix + 'noise_f64'], noise)
        assert np.array_equal(data[prefix + 'mean_before'], mean) and np.array_equal(data[prefix + 'std_before'], std)
        sampled = (mean[:, None] + std[:, None] * noise).astype(np.float32)
        u = np.clip(unnorm(np, sampled), -1, 1)
        u[:, 0] = 0.
        u[:, 1] = warm
        z = norm(np, u)
        shape_dtype(np, data[prefix + 'actual_u'], u.shape, np.float32)
        shape_dtype(np, data[prefix + 'actual_z'], z.shape, np.float32)
        assert np.array_equal(data[prefix + 'actual_u'], u) and np.array_equal(data[prefix + 'actual_z'], z)
        terminal = data[prefix + 'terminal_E']
        shape_dtype(np, terminal, (8, 64, 192), np.float32)
        cost = ((terminal[:, :, None] - data['goal_E'][None, None]) ** 2).mean(-1).min(-1)
        assert cost.dtype == np.float32 and np.array_equal(data[prefix + 'cost'], cost)
        elite = np.argsort(cost, axis=1, kind='stable')[:, :8]
        assert np.array_equal(data[prefix + 'elite'], elite)
        for slot in range(8):
            chosen = int(np.argmin(cost[slot]))
            # Strict improvement preserves the lower global ID on exact ties.
            if cost[slot, chosen] < best_cost[slot]:
                best_cost[slot] = cost[slot, chosen]
                best_id[slot] = iteration * 64 + chosen
                best_u[slot] = u[slot, chosen]
            postclip_elites = z[slot, elite[slot]]
            mean[slot] = .25 * mean[slot] + .75 * postclip_elites.mean(0)
            std[slot] = np.maximum(.1, .25 * std[slot] + .75 * postclip_elites.std(0))
        assert np.array_equal(data[prefix + 'mean_after'], mean) and np.array_equal(data[prefix + 'std_after'], std)
    assert np.array_equal(data['selected_u'], best_u)
    assert np.array_equal(data['selected_global_id'], best_id) and np.array_equal(data['selected_cost'], best_cost)
    assert np.array_equal(data['selected_first5_u'], np.repeat(best_u[:, 0, None, :], 5, axis=1))
    assert np.array_equal(data['shifted_warm_u'], np.concatenate((best_u[:, 1:], best_u[:, -1:]), axis=1))

    return best_u
