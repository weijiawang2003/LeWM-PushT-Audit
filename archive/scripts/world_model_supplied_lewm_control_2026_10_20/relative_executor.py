"""Root-only relative native actuation and fixed engineering parity fixture.

Import defines stdlib-only helpers. The caller supplies the already qualified
NumPy/native runtime. A controller must receive only observation96() copies and
its own relative-action history, never this object or its private driver API.
The driver owns independent per-raw coverage/first-hit checks and durable logs.
"""
import copy
import hashlib
import inspect
import math
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]
GOAL = (256.0, 256.0, math.pi / 4)
MAX_RAW_CONTROLS = 300
ACTION_SCALE = 100.0
SOURCE_BINDINGS = {
    "src/jepa4object/environment.py":
        "23446116ccfbff22ea9a76639ae2a90b489b0ba2a57526bf2b36594ddb417203",
    ".venv_object/lib/python3.12/site-packages/gym_pusht/envs/pusht.py":
        "8d3b6b7d065e46fb2bf945b196068cd6c96c107e9644b2606f5ac4d8e7452ff2",
}
# Fixed before any outcomes. Boundary poses make the first issued target leave
# the old absolute [0,512] domain; neither pose is an evaluation/closed test.
FIXTURE_CASES = (
    (97000, (32.0, 80.0, 256.0, 256.0, 0.0), -1.0),
    (97001, (480.0, 432.0, 256.0, 256.0, 0.0), 1.0),
)
FIXTURE_U = (
    (1., 0.), (.25, .5), (-.5, -.25), (0., -.75), (.75, .25),
    (-.25, .75), (.5, -.5), (-.75, 0.), (.25, -.25), (0., .5),
    (-.5, .25), (.75, -.75), (-.25, -.5), (.5, .75), (0., 0.),
)
FIXTURE_PLAN = {
    "schema": "RELATIVE_NATIVE_PARITY_FIXTURE_v1",
    "seeds": [97000, 97001], "raw_controls_per_sequence": 15,
    "paired_native_sequences": 4, "own_relative_replay_sequences": 2,
    "native_raw_controls": 90, "expected_physics_substeps": 900,
    "render_calls": 4, "invalid_actions_rejected_before_physics": 8,
    "state11_comparison": "float64 numerical elementwise exact, not byte identity",
    "not_control_performance_or_training": True,
    "closed_seeds_99000_to_99015_not_accessed": True,
}


def source_bindings(root=ROOT):
    root = Path(root).resolve()
    observed = {}
    for name, expected in SOURCE_BINDINGS.items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("qualified native source unavailable")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("qualified native source changed")
        observed[name] = actual
    return observed


def _check_base(base, root):
    expected = list(SOURCE_BINDINGS)
    if (type(base).__module__, type(base).__name__) != ("jepa4object.environment", "PushTStateEnv"):
        raise TypeError("exact qualified PushTStateEnv required")
    if (type(base.env).__module__, type(base.env).__name__) != ("gym_pusht.envs.pusht", "PushTEnv"):
        raise TypeError("exact qualified native PushTEnv required")
    for obj, name in ((base, expected[0]), (base.env, expected[1])):
        if Path(inspect.getfile(type(obj))).resolve() != (root / name).resolve():
            raise ValueError("native class source path differs")
    native = base.env
    if (native.dt, native.control_hz, native.k_p, native.k_v) != (.01, 10, 100, 20):
        raise ValueError("native PD/control contract differs")
    if int(1 / (native.dt * native.control_hz)) != 10 or native.obs_type != "state":
        raise ValueError("native substep/observation contract differs")


class RelativeExecutor:
    """Driver-private wrapper; no state, native return or info is a step output.

    Supply a durable private_sink(event) in an actual controller run. A started
    event precedes each native call, so incomplete native execution is distinct
    from a returned control. This source performs no policy decision or success
    gating. The driver must stop at its verified first hit and verify every raw
    state; the wrapper independently refuses a 301st control.
    """
    def __init__(self, np, base, private_sink=None, *, root=ROOT):
        self._np, self._base = np, base
        self._root = Path(root).resolve()
        self._bindings = source_bindings(self._root)
        _check_base(base, self._root)
        if private_sink is not None and not callable(private_sink):
            raise TypeError("private sink must be callable")
        self._sink = private_sink
        self._events, self._controls, self._relative_actions = [], [], []
        self._ready = self._faulted = self._closed = False
        self._session = -1
        self._seed = None

    def _emit(self, event):
        saved = copy.deepcopy(event)
        self._events.append(saved)
        if self._sink is not None:
            self._sink(copy.deepcopy(saved))

    def _state11(self):
        value = self._np.asarray(self._base.state(), dtype=self._np.float64).copy()
        if value.shape != (11,) or not bool(self._np.isfinite(value).all()):
            raise ValueError("native state11 is invalid")
        return value

    def _action(self, u):
        value = self._np.asarray(u, dtype=self._np.float64).copy()
        if value.shape != (2,) or not bool(self._np.isfinite(value).all()):
            raise ValueError("relative action must be finite shape(2)")
        if bool((value < -1.).any()) or bool((value > 1.).any()):
            raise ValueError("relative action must be in [-1,1]")
        return value

    def reset(self, *, seed, requested_pose=None, goal=GOAL):
        if self._closed or self._faulted:
            raise RuntimeError("closed/faulted executor cannot reset or retry")
        np = self._np
        requested = None if requested_pose is None else np.asarray(requested_pose, dtype=np.float64)
        if requested is not None and (requested.shape != (5,) or not bool(np.isfinite(requested).all())):
            raise ValueError("requested reset pose must be finite shape(5)")
        fixed_goal = np.asarray(goal, dtype=np.float64)
        if fixed_goal.shape != (3,) or not bool(np.array_equal(fixed_goal, np.asarray(GOAL))):
            raise ValueError("this executor is fixed to the public T goal")
        self._session += 1
        self._seed = int(seed)
        self._ready = False
        self._relative_actions = []
        self._emit({"event": "relative_reset_started", "session": self._session, "seed": self._seed,
                    "requested_pose": None if requested is None else requested.tolist()})
        try:
            self._base.reset(seed=self._seed, requested_pose=requested, goal=fixed_goal)
            # The old ReplaySpec contains absolute targets; it is inapplicable.
            self._base.current_replay = None
            if self._base.control_steps != 0:
                raise ValueError("native reset did not clear control count")
            if not bool(np.array_equal(np.asarray(self._base.env.goal_pose), fixed_goal)):
                raise ValueError("native goal differs from verified public goal")
            state = self._state11()
            self._ready = True
            self._emit({"event": "relative_reset_returned", "session": self._session, "seed": self._seed,
                        "actual_state11": state.tolist(), "control_steps": 0,
                        "legacy_absolute_replay_disabled": True})
        except Exception as error:
            self._faulted = True
            self._emit({"event": "relative_reset_failed", "session": self._session,
                        "error_type": type(error).__name__})
            raise

    def step(self, u):
        if not self._ready or self._faulted or self._closed:
            raise RuntimeError("executor must be reset, live and not faulted")
        issued = self._action(u)
        raw_t = self._base.control_steps
        if type(raw_t) is not int or raw_t != len(self._relative_actions) or not 0 <= raw_t < MAX_RAW_CONTROLS:
            raise RuntimeError("relative control count/budget contract")
        cpu, wall = time.process_time(), time.perf_counter()
        before = self._state11()
        # Snapshot before execution. No clamp and no blockwise target reuse.
        target = before[:2] + ACTION_SCALE * issued
        row = {"session": self._session, "seed": self._seed, "raw_t_before": raw_t,
               "issued_u": issued.tolist(), "actual_target": target.tolist(),
               "before_state11": before.tolist(), "after_state11": None,
               "target_outside_old_absolute_box": bool(((target < 0.) | (target > 512.)).any()),
               "native_step_returned": False, "control_steps_after": raw_t,
               "expected_native_step_calls": 1, "native_step_calls_started": 0,
               "expected_native_substeps": 10,
               "no_absolute_target_clip": True, "not_a_planner_observation": True}
        self._emit({"event": "relative_control_started", **row})
        try:
            row["native_step_calls_started"] = 1
            self._base.env.step(target)
            row["native_step_returned"] = True
            self._base.control_steps += 1
            self._relative_actions.append(issued.tolist())
            row["control_steps_after"] = self._base.control_steps
            row["after_state11"] = self._state11().tolist()
        except Exception as error:
            self._faulted = True
            row["error_type"] = type(error).__name__
            row["unreturned_native_call_may_have_advanced_physics"] = not row["native_step_returned"]
            try:
                row["after_state11"] = self._state11().tolist()
            except Exception:
                pass
            raise
        finally:
            row["body_CPU_s_nested"] = time.process_time() - cpu
            row["body_wall_s_nested"] = time.perf_counter() - wall
            row["timing_excludes_final_sink_and_is_not_outer_fee"] = True
            self._controls.append(copy.deepcopy(row))
            self._emit({"event": "relative_control_closed", **row})
        return None

    def observation96(self):
        if not self._ready or self._faulted or self._closed:
            raise RuntimeError("executor observation unavailable")
        np = self._np
        image = np.ascontiguousarray(self._base.env._render(visualize=False)).copy()
        if image.shape != (96, 96, 3) or image.dtype != np.uint8:
            raise ValueError("qualified RGB96 observation contract")
        return image

    def replay_relative(self, *, seed, actions, requested_pose=None, goal=GOAL):
        values = self._np.asarray(actions, dtype=self._np.float64)
        if values.ndim != 2 or values.shape[1] != 2 or len(values) > MAX_RAW_CONTROLS:
            raise ValueError("bounded relative replay sequence required")
        fixed = [self._action(u) for u in values]
        self.reset(seed=seed, requested_pose=requested_pose, goal=goal)
        for u in fixed:
            self.step(u)
        return None

    # Driver/evaluator-only access. Never pass this object or these outputs to
    # the image-only controller. Python naming is not an OS/process sandbox.
    def private_state11(self):
        return self._state11()

    def private_records(self):
        return copy.deepcopy(self._controls)

    def private_events(self):
        return copy.deepcopy(self._events)

    def close(self):
        if not self._closed:
            self._base.close()
            self._closed = True


def run_parity_fixture(np, env_factory, private_sink=None, *, root=ROOT):
    """Fixed 90-control qualification, no control effect/model selection.

    Two native environments per seed: wrapper versus an independently written
    target equation/native call. Then reset/replay only the relative commands.
    The shared native simulator/state11 extractor is explicitly not an
    independent physics implementation. The root guard charges the full call.
    """
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    bindings = source_bindings(root)
    cases = []
    for seed, requested, first_x in FIXTURE_CASES:
        wrapper_base, reference = env_factory(), env_factory()
        executor = None
        try:
            executor = RelativeExecutor(np, wrapper_base, private_sink, root=root)
            _check_base(reference, Path(root).resolve())
            executor.reset(seed=seed, requested_pose=requested)
            reference.reset(seed=seed, requested_pose=requested, goal=GOAL)
            reference.current_replay = None
            initial = executor.private_state11()
            assert bool(np.array_equal(initial, reference.state()))
            assert executor.observation96().shape == (96, 96, 3)
            rejected = []
            for bad in ((float("nan"), 0.), (0., float("inf")), (0., 0., 0.), (1.01, 0.)):
                count = wrapper_base.control_steps
                state = executor.private_state11()
                try:
                    executor.step(bad)
                except ValueError:
                    rejected.append(True)
                else:
                    raise AssertionError("invalid relative action reached native physics")
                assert wrapper_base.control_steps == count and bool(np.array_equal(state, executor.private_state11()))
            actions = [list(u) for u in FIXTURE_U]
            actions[0][0] = first_x
            rows, original_states = [], []
            for raw_t, u in enumerate(actions):
                previous = np.asarray(reference.state(), dtype=np.float64).copy()
                p = reference.env.agent.position
                # Independent equation; does not call the wrapper/target helper.
                target = np.asarray((float(p.x) + 100. * float(u[0]),
                                     float(p.y) + 100. * float(u[1])), dtype=np.float64)
                returned = executor.step(u)
                assert returned is None
                reference.env.step(target)
                reference.control_steps += 1
                actual = executor.private_state11()
                truth = np.asarray(reference.state(), dtype=np.float64).copy()
                saved = executor.private_records()[-1]
                assert bool(np.array_equal(actual, truth))
                assert saved["before_state11"] == previous.tolist()
                assert saved["actual_target"] == target.tolist()
                assert wrapper_base.control_steps == reference.control_steps == raw_t + 1
                assert wrapper_base.current_replay is None and reference.current_replay is None
                original_states.append(actual.tolist())
                rows.append({"raw_t": raw_t + 1, "u": list(u), "reference_target": target.tolist(),
                             "reference_before_state11": previous.tolist(), "reference_after_state11": truth.tolist(),
                             "wrapper": saved, "state11_elementwise_exact": True})
            assert rows[0]["wrapper"]["target_outside_old_absolute_box"]
            assert executor.observation96().shape == (96, 96, 3)
            first_record_count = len(executor.private_records())
            executor.replay_relative(seed=seed, actions=actions, requested_pose=requested)
            replay = executor.private_records()[first_record_count:]
            assert len(replay) == 15 and [r["after_state11"] for r in replay] == original_states
            cases.append({"seed": seed, "requested_pose": list(requested), "initial_state11": initial.tolist(),
                          "actions": actions, "invalid_action_rejections": len(rejected), "rows": rows,
                          "own_relative_replay_records": replay, "private_events": executor.private_events(),
                          "out_of_box_target_exercised": True, "passed": True})
        finally:
            if executor is not None:
                executor.close()
            else:
                wrapper_base.close()
            reference.close()
    return {"schema": "RELATIVE_NATIVE_PARITY_RESULT_v1", "passed": True,
            "plan": copy.deepcopy(FIXTURE_PLAN), "source_bindings": bindings, "cases": cases,
            "same_native_backend_not_independent_physics": True, "no_old_absolute_restore_or_replay": True,
            "no_model_training_task_policy_or_closed_test": True,
            "no_policy_performance_selection": True,
            "fixture_body_CPU_s_nested": time.process_time() - started_cpu,
            "fixture_body_wall_s_nested": time.perf_counter() - started_wall,
            "body_timing_is_not_complete_outer_guard_fee": True}
