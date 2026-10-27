"""Root-only fixed-B8 execution driver; source definitions, no backend loader.

Caller must separately admit control after full qualification and complete-cost
forecast. Inject the qualified NumPy/native/executor/policy objects. The only
policy call is decide(eight copied RGB96 arrays); verifier data never enters it.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(88000, 88008))
GOAL = (256., 256., math.pi / 4)
MAX_RAW, FEEDBACK, MAX_ROUNDS = 300, 5, 60
COVERAGE_TOLERANCE = 1e-8


class PanelStop(Exception):
    """Normal finite-resource stop; never interpreted as task failure0/8."""


class Phases:
    def __init__(self):
        self.rows = {}

    @contextmanager
    def phase(self, name):
        cpu, wall = time.process_time(), time.perf_counter()
        try:
            yield
        finally:
            row = self.rows.setdefault(name, {"calls": 0, "CPU_s_nested": 0., "wall_s_nested": 0.})
            row["calls"] += 1
            row["CPU_s_nested"] += time.process_time() - cpu
            row["wall_s_nested"] += time.perf_counter() - wall


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def _append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def _finite_or_none(value):
    return value if value is not None and math.isfinite(value) else None


def run_panel(np, env_factory, coverage_from_state, policy, outdir, *, executor_class,
              deadline_unix, cpu_budget_s=180., wall_budget_s=900.,
              phase_cpu_origin=None, phase_wall_origin=None,
              reserve_CPU_s=2., reserve_wall_s=5., resource_stop=None, metadata=None):
    """Run only after an external qualification/forecast/admission guard.

    policy.decide(images8) returns (float32 u[8,5,2], JSON-safe receipt). It owns
    its round/RNG/warm state and policy-query artifact directory. The caller may
    precreate outdir/policy_queries and load the model there before this call;
    all driver-owned paths are exclusive. That cold load is outside this
    function clock, but remains inside the required outer guard's full fee.

    Normal resource stops return a partial RESULT with no complete-panel rate.
    Unexpected errors return FAILURE. An outer kill may prevent a terminal
    receipt; PANEL_STARTED and durable prefixes then explicitly mean partial.
    This function creates no CLI, dependency loader, admission or retry route.
    """
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    cpu_origin = start_cpu if phase_cpu_origin is None else float(phase_cpu_origin)
    wall_origin = start_wall if phase_wall_origin is None else float(phase_wall_origin)
    for value in (deadline_unix, cpu_budget_s, wall_budget_s, reserve_CPU_s, reserve_wall_s,
                  cpu_origin, wall_origin):
        if not math.isfinite(float(value)):
            raise ValueError("finite resource clocks required")
    if not (0 <= reserve_CPU_s < cpu_budget_s and 0 <= reserve_wall_s < wall_budget_s):
        raise ValueError("positive budget with cleanup reserve required")
    if not callable(getattr(policy, "decide", None)):
        raise TypeError("policy.decide(images8) required")
    out = Path(outdir).resolve()
    out.relative_to(ROOT / "results/world_model_supplied_lewm_control_2026_10_20")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("PANEL_STARTED.json", "DRIVER_CONTRACT.json", "RESULT.json", "FAILURE.json",
                 "PRIVATE_EXECUTOR_EVENTS.jsonl", "TRANSITIONS.jsonl", "EPISODES.jsonl",
                 "ROUNDS.jsonl", "COSTS.json"):
        if (out / name).exists():
            raise FileExistsError("driver output already exists")
    if any((out / ("episode_" + str(seed) + ".npz")).exists() for seed in SEEDS):
        raise FileExistsError("episode output already exists")
    for name in ("PRIVATE_EXECUTOR_EVENTS.jsonl", "TRANSITIONS.jsonl", "EPISODES.jsonl", "ROUNDS.jsonl"):
        (out / name).open("x").close()
    (out / "decision_receipts").mkdir(exist_ok=False)
    _write(out / "PANEL_STARTED.json", {"schema": "SUPPLIED_LEWM_PANEL_STARTED_v1",
        "seeds": list(SEEDS), "status_until_terminal_receipt": "partial_unqualified",
        "missing_terminal_is_not_zero_success_rate": True,
        "outer_guard_required": True, "started_unix": time.time()})
    _write(out / "DRIVER_CONTRACT.json", {"schema": "SUPPLIED_LEWM_DRIVER_CONTRACT_v1",
        "seeds": list(SEEDS), "fixed_batch_size": 8, "max_raw_controls": MAX_RAW,
        "feedback_raw_controls": FEEDBACK, "max_group_rounds": MAX_ROUNDS, "goal": list(GOAL),
        "policy_arguments": "only eight RGB96 copies; no active mask/state/info/coverage/executor",
        "inactive_inputs": "last cached image; still predicted and never executed",
        "success_predicate": "native and independent coverage strictly greater than0.95",
        "coverage_tolerance": COVERAGE_TOLERANCE,
        "relative_target": "private current pusher position +100*u, no absolute target clip",
        "driver_does_not_clip_or_renormalize_policy_u": True,
        "source_call_boundary_not_OS_sandbox": True,
        "no_model_loader_or_new_policy_in_this_driver": True,
        "caller_metadata": {} if metadata is None else metadata})
    phases = Phases()
    episodes = [{"slot": i, "seed": seed, "base": None, "executor": None,
        "initialized": False, "close_attempted": False, "closed": False,
        "states": [], "state_raw_t": [], "native": [], "independent": [], "verified": [],
        "u": [], "target": [], "cache": None, "first_hit": None,
        "complete": False, "artifact_attempted": False, "artifact_saved": False,
        "trajectory": None, "active_decisions": 0, "last_closed_control": None,
        "pending_native_attempt": None} for i, seed in enumerate(SEEDS)]
    rounds, errors, stop_reason = [], [], None
    policy_calls_started = policy_calls_returned = 0
    method_error = None

    def check_stop():
        if time.process_time() - cpu_origin >= cpu_budget_s - reserve_CPU_s:
            raise PanelStop("fixed_CPU_budget_reserve")
        if time.perf_counter() - wall_origin >= wall_budget_s - reserve_wall_s:
            raise PanelStop("fixed_wall_budget_reserve")
        if time.time() >= float(deadline_unix) - reserve_wall_s:
            raise PanelStop("fixed_absolute_deadline_reserve")
        if resource_stop is not None:
            reason = resource_stop()
            if reason:
                raise PanelStop("external_resource_stop")

    def sink(slot, event):
        ep = episodes[slot]
        if event.get("event") == "relative_control_closed":
            ep["last_closed_control"] = event
        with phases.phase("private_executor_event_serialization"):
            _append(out / "PRIVATE_EXECUTOR_EVENTS.jsonl", {"slot": slot, "episode_seed": ep["seed"], **event})

    def snapshot(ep, issued=None, target=None):
        raw_t = len(ep["u"])
        with phases.phase("private_state_native_independent_verification"):
            state = np.asarray(ep["executor"].private_state11(), dtype=np.float64).copy()
            if state.shape != (11,) or not bool(np.isfinite(state).all()):
                raise ValueError("invalid actual state11")
            if ep["state_raw_t"] and ep["state_raw_t"][-1] >= raw_t:
                raise ValueError("state raw clock duplicated")
            ep["states"].append(state); ep["state_raw_t"].append(raw_t)
            ep["native"].append(None); ep["independent"].append(None); ep["verified"].append(False)
            row = {"slot": ep["slot"], "episode_seed": ep["seed"], "raw_t": raw_t,
                   "state11": state.tolist(), "issued_u": issued, "actual_target": target,
                   "native_coverage": None, "independent_coverage": None, "verified": False}
            try:
                native = float(ep["base"].env._get_coverage())
                independent = float(coverage_from_state(state.copy(), np.asarray(GOAL, dtype=np.float64)))
                ep["native"][-1], ep["independent"][-1] = native, independent
                row["native_coverage"], row["independent_coverage"] = _finite_or_none(native), _finite_or_none(independent)
                if not (math.isfinite(native) and math.isfinite(independent)
                        and -COVERAGE_TOLERANCE <= native <= 1. + COVERAGE_TOLERANCE
                        and -COVERAGE_TOLERANCE <= independent <= 1. + COVERAGE_TOLERANCE
                        and abs(native - independent) < COVERAGE_TOLERANCE):
                    raise ValueError("native independent coverage disagreement")
                if (native > .95) != (independent > .95):
                    raise ValueError("strict success threshold disagreement")
                ep["verified"][-1] = True; row["verified"] = True
                row["success_gt_0_95"] = native > .95
                if native > .95 and ep["first_hit"] is None:
                    ep["first_hit"] = raw_t
            except Exception as error:
                row["verification_error_type"] = type(error).__name__
                raise
            finally:
                with phases.phase("raw_transition_serialization"):
                    _append(out / "TRANSITIONS.jsonl", row)

    def close_episode(ep):
        if ep["base"] is not None and not ep["close_attempted"]:
            ep["close_attempted"] = True
            with phases.phase("native_close"):
                if ep["executor"] is not None:
                    ep["executor"].close()
                else:
                    ep["base"].close()
            ep["closed"] = True

    def summary(ep):
        return {"slot": ep["slot"], "seed": ep["seed"], "initialized": ep["initialized"],
            "complete": ep["complete"], "success": ep["first_hit"] is not None,
            "first_hit_raw_t": ep["first_hit"], "returned_raw_controls": len(ep["u"]),
            "observed_states": len(ep["states"]), "verified_states": sum(ep["verified"]),
            "active_decisions": ep["active_decisions"], "closed": ep["closed"],
            "trajectory": ep["trajectory"], "artifact_saved": ep["artifact_saved"],
            "pending_native_attempt": ep["pending_native_attempt"],
            "initial_coverage": _finite_or_none(ep["native"][0]) if ep["native"] else None,
            "final_coverage": _finite_or_none(ep["native"][-1]) if ep["native"] else None,
            "partial_is_not_a_complete_panel_failure": not ep["complete"]}

    def save_episode(ep):
        if ep["artifact_attempted"]:
            return
        ep["artifact_attempted"] = True
        path = out / ("episode_" + str(ep["seed"]) + ".npz")
        with phases.phase("episode_trajectory_serialization"):
            with path.open("xb") as stream:
                np.savez(stream, state11=np.asarray(ep["states"], dtype=np.float64).reshape(-1, 11),
                    state_raw_t=np.asarray(ep["state_raw_t"], dtype=np.int64),
                    state_verified=np.asarray(ep["verified"], dtype=np.bool_),
                    issued_u=np.asarray(ep["u"], dtype=np.float32).reshape(-1, 2),
                    actual_target=np.asarray(ep["target"], dtype=np.float64).reshape(-1, 2),
                    native_coverage=np.asarray([np.nan if x is None else x for x in ep["native"]], dtype=np.float64),
                    independent_coverage=np.asarray([np.nan if x is None else x for x in ep["independent"]], dtype=np.float64),
                    goal=np.asarray(GOAL, dtype=np.float64))
                stream.flush(); os.fsync(stream.fileno())
            ep["trajectory"] = {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha(path)}
            ep["artifact_saved"] = True
            _append(out / "EPISODES.jsonl", summary(ep))

    def finish(ep):
        count = len(ep["u"])
        if not (ep["first_hit"] is not None or count == MAX_RAW):
            raise ValueError("episode has no verified terminal or budget endpoint")
        if len(ep["states"]) != count + 1 or ep["state_raw_t"] != list(range(count + 1)) or not all(ep["verified"]):
            raise ValueError("complete episode lacks full raw verification")
        hits = [i for i, value in enumerate(ep["native"]) if value > .95]
        if (min(hits) if hits else None) != ep["first_hit"]:
            raise ValueError("first-hit mismatch")
        close_episode(ep)
        ep["complete"] = True
        save_episode(ep)

    def execute_one(ep, u):
        before_count = len(ep["u"])
        error = None
        try:
            with phases.phase("relative_native_control_including_private_logs"):
                ep["executor"].step(u)
        except Exception as caught:
            error = caught
        record = ep["last_closed_control"]
        if record is not None and record["raw_t_before"] == before_count:
            if record["native_step_returned"]:
                if record["control_steps_after"] != before_count + 1:
                    raise ValueError("returned native control count mismatch")
                ep["u"].append(list(record["issued_u"]))
                ep["target"].append(list(record["actual_target"]))
                if record["before_state11"] != ep["states"][-1].tolist():
                    raise ValueError("native transition continuity mismatch")
                expected = np.asarray(record["before_state11"][:2]) + 100. * np.asarray(record["issued_u"])
                if not bool(np.array_equal(expected, np.asarray(record["actual_target"]))):
                    raise ValueError("relative target log mismatch")
                if record["issued_u"] != np.asarray(u, dtype=np.float64).tolist():
                    raise ValueError("executed relative command differs from selected command")
                snapshot(ep, record["issued_u"], record["actual_target"])
                if record["after_state11"] != ep["states"][-1].tolist():
                    raise ValueError("private post-state log mismatch")
            else:
                ep["pending_native_attempt"] = record
        elif error is None:
            raise ValueError("returned control lacks durable closed record")
        if error is not None:
            raise error

    try:
        for ep in episodes:
            check_stop()
            with phases.phase("native_construct_reset"):
                ep["base"] = env_factory()
                ep["executor"] = executor_class(np, ep["base"], private_sink=lambda event, slot=ep["slot"]: sink(slot, event))
                ep["executor"].reset(seed=ep["seed"], goal=GOAL)
                ep["initialized"] = True
            snapshot(ep)
        if all(ep["first_hit"] is not None for ep in episodes):
            for ep in episodes:
                finish(ep)
        for round_index in range(MAX_ROUNDS):
            if all(ep["complete"] for ep in episodes):
                break
            check_stop()
            round_cpu, round_wall = time.process_time(), time.perf_counter()
            active = [ep["slot"] for ep in episodes if not ep["complete"] and ep["first_hit"] is None]
            # The first group also creates a cache for a possible t0-success
            # slot before closing it. Later inactive slots are never rendered.
            for ep in episodes:
                if ep["slot"] in active or ep["cache"] is None:
                    with phases.phase("RGB96_render_and_cache"):
                        ep["cache"] = ep["executor"].observation96().copy()
                if ep["first_hit"] is not None and not ep["complete"]:
                    finish(ep)
            if not active:
                break
            images = np.stack([ep["cache"].copy() for ep in episodes])
            if images.shape != (8, 96, 96, 3) or images.dtype != np.uint8:
                raise ValueError("fixed B8 RGB input invalid")
            image_hashes = [hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest() for image in images]
            _append(out / "ROUNDS.jsonl", {"event": "round_started", "round": round_index,
                "active_slots_private": active, "fixed_policy_batch": 8, "RGB_sha256": image_hashes})
            with phases.phase("full_fixed_B8_policy_query"):
                policy_calls_started += 1
                chunks, receipt = policy.decide(images.copy())
                policy_calls_returned += 1
            chunks = np.asarray(chunks)
            if chunks.shape != (8, 5, 2) or chunks.dtype != np.float32 or not bool(np.isfinite(chunks).all()):
                raise ValueError("policy must return finite float32 B8x5x2 actual u")
            if bool((chunks < -1.).any()) or bool((chunks > 1.).any()):
                raise ValueError("policy command outside fixed relative domain")
            chunks = chunks.copy()
            # JSON encoding also rejects nonfinite/opaque receipt values.
            json.dumps(receipt, allow_nan=False)
            receipt_path = out / "decision_receipts" / ("round_" + str(round_index).zfill(3) + ".json")
            with phases.phase("decision_receipt_serialization"):
                _write(receipt_path, {"round": round_index, "fixed_policy_batch": 8,
                    "RGB_sha256": image_hashes, "actual_u_8x5x2": chunks.tolist(), "policy_receipt": receipt})
                receipt_binding = {"path": str(receipt_path.relative_to(out)), "bytes": receipt_path.stat().st_size,
                                   "sha256": _sha(receipt_path)}
            for slot in active:
                episodes[slot]["active_decisions"] += 1
            before_execution_counts = [len(ep["u"]) for ep in episodes]
            round_stop = None
            try:
                check_stop()
                for offset in range(FEEDBACK):
                    for slot in active:
                        ep = episodes[slot]
                        if ep["complete"]:
                            continue
                        check_stop()
                        execute_one(ep, chunks[slot, offset])
                        if ep["first_hit"] is not None or len(ep["u"]) == MAX_RAW:
                            finish(ep)
            except PanelStop as error:
                round_stop = str(error)
                raise
            finally:
                row = {"event": "round_closed", "round": round_index, "active_slots_private": active,
                    "fixed_policy_batch": 8, "RGB_sha256": image_hashes, "decision_receipt": receipt_binding,
                    "executed_raw_controls_per_slot": [len(ep["u"]) - before_execution_counts[i]
                                                        for i, ep in enumerate(episodes)],
                    "inactive_predictions_never_executed": True,
                    "finished_slots": [slot for slot in active if episodes[slot]["complete"]],
                    "resource_stop_reason": round_stop,
                    "full_group_CPU_s_nested": time.process_time() - round_cpu,
                    "full_group_wall_s_nested": time.perf_counter() - round_wall,
                    "group_includes_render_query_receipts_execution_verification_terminal_serialization": True,
                    "not_B1_latency": True}
                rounds.append(row); _append(out / "ROUNDS.jsonl", row)
        if not all(ep["complete"] for ep in episodes):
            raise ValueError("60 group rounds ended without a complete or explicitly stopped panel")
    except PanelStop as error:
        stop_reason = str(error)
    except Exception as error:
        method_error = type(error).__name__
        stop_reason = "driver_or_policy_or_verification_error"
    finally:
        for ep in episodes:
            try:
                close_episode(ep)
            except Exception as error:
                errors.append({"slot": ep["slot"], "phase": "close", "error_type": type(error).__name__})
            try:
                save_episode(ep)
            except Exception as error:
                errors.append({"slot": ep["slot"], "phase": "save_episode", "error_type": type(error).__name__})
    complete = (all(ep["complete"] and ep["artifact_saved"] and ep["closed"] for ep in episodes)
                and stop_reason is None and not errors)
    with phases.phase("cost_ledger_serialization"):
        _write(out / "COSTS.json", {"phases": phases.rows,
            "phase_clocks_are_nested_and_not_summed_into_full_fee": True,
            "policy_receipt_clocks_nested_inside_policy_phase": True,
            "model_import_load_outside_function_body_remains_outer_guard_cost": True,
            "final_metadata_hash_terminal_write_and_exit_remain_outer_guard_cost": True})
    files = {str(path.relative_to(out)): {"bytes": path.stat().st_size, "sha256": _sha(path)}
             for path in sorted(out.rglob("*")) if path.is_file()}
    result = {"schema": "SUPPLIED_LEWM_FIXED_B8_PANEL_RESULT_v1", "passed": complete,
        "panel_complete": complete, "partial": not complete, "stop_reason": stop_reason,
        "error_type": method_error, "cleanup_errors": errors,
        "episodes": [summary(ep) for ep in episodes], "rounds": rounds,
        "complete_panel_successes": sum(ep["first_hit"] is not None for ep in episodes) if complete else None,
        "complete_panel_success_rate": sum(ep["first_hit"] is not None for ep in episodes) / 8. if complete else None,
        "observed_successes_in_prefix_not_a_panel_rate": sum(ep["first_hit"] is not None for ep in episodes),
        "returned_raw_controls": sum(len(ep["u"]) for ep in episodes),
        "policy_group_calls_started": policy_calls_started,
        "policy_group_calls_returned": policy_calls_returned,
        "saved_decision_receipts": len(list((out / "decision_receipts").glob("*.json"))),
        "fixed_B8_prediction_opportunity_including_inactive_slots": True,
        "policy_received_state_info_coverage_active_mask_or_executor": False,
        "no_native_future_calls_for_selection": True, "one_fixed_goal_not_cross_goal_reuse": True,
        "public_reused_validation_not_new_confirmatory_test": True,
        "source_call_boundary_not_OS_sandbox": True, "artifacts_before_terminal": files,
        "function_CPU_s_before_terminal": time.process_time() - start_cpu,
        "function_wall_s_before_terminal": time.perf_counter() - start_wall,
        "function_clock_excludes_external_model_load_and_terminal_write": True,
        "outer_guard_required_for_complete_fee": True,
        "missing_terminal_or_hard_kill_is_partial_not_zero_success_rate": True}
    _write(out / ("FAILURE.json" if method_error is not None or errors else "RESULT.json"), result)
    return result
