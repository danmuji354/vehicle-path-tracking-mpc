"""Kinematic bicycle model and constrained nonlinear MPC."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np
from scipy.optimize import minimize

Array = np.ndarray


@dataclass(frozen=True)
class VehicleParams:
    wheelbase_m: float = 2.7
    max_steer_rad: float = 0.48
    max_accel_mps2: float = 2.0
    min_accel_mps2: float = -3.0
    max_speed_mps: float = 12.0


def wrap_angle(angle: float | Array) -> float | Array:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def reference_at(x: float) -> Array:
    y = 1.5 * np.sin(x / 9.0)
    dy_dx = (1.5 / 9.0) * np.cos(x / 9.0)
    yaw = np.arctan(dy_dx)
    return np.array([x, y, yaw, 8.0])


class Bicycle:
    def __init__(self, params: VehicleParams | None = None):
        self.params = VehicleParams() if params is None else params

    def derivative(self, state: Array, control: Array, friction_scale: float = 1.0) -> Array:
        _, _, yaw, speed = np.asarray(state, dtype=float)
        accel = float(np.clip(control[0], self.params.min_accel_mps2, self.params.max_accel_mps2))
        steer_limit = self.params.max_steer_rad * friction_scale
        steer = float(np.clip(control[1], -steer_limit, steer_limit))
        return np.array(
            [speed * np.cos(yaw), speed * np.sin(yaw), speed * np.tan(steer) / self.params.wheelbase_m, accel]
        )

    def step(self, state: Array, control: Array, dt: float, friction_scale: float = 1.0) -> Array:
        f = lambda s: self.derivative(s, control, friction_scale)
        k1 = f(state)
        k2 = f(state + dt * k1 / 2)
        k3 = f(state + dt * k2 / 2)
        k4 = f(state + dt * k3)
        nxt = np.asarray(state) + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        nxt[2] = wrap_angle(nxt[2])
        nxt[3] = np.clip(nxt[3], 0, self.params.max_speed_mps)
        return nxt


class PurePursuit:
    def __init__(self, plant: Bicycle, lookahead_m: float = 5.0):
        self.plant = plant
        self.lookahead_m = lookahead_m

    def command(self, state: Array) -> Array:
        lookahead_x = state[0] + self.lookahead_m
        target = reference_at(lookahead_x)
        alpha = wrap_angle(np.arctan2(target[1] - state[1], target[0] - state[0]) - state[2])
        steer = np.arctan2(2 * self.plant.params.wheelbase_m * np.sin(alpha), self.lookahead_m)
        accel = 1.2 * (target[3] - state[3])
        return np.array([np.clip(accel, -3, 2), np.clip(steer, -0.48, 0.48)])


class LateralLQR:
    """Gain-scheduled linear error feedback baseline."""

    def __init__(self, plant: Bicycle):
        self.plant = plant

    def command(self, state: Array) -> Array:
        ref = reference_at(state[0])
        error_y = state[1] - ref[1]
        error_yaw = wrap_angle(state[2] - ref[2])
        steer = -0.34 * error_y - 1.15 * error_yaw
        accel = 1.0 * (ref[3] - state[3])
        return np.array([np.clip(accel, -3, 2), np.clip(steer, -0.48, 0.48)])


class NonlinearMPC:
    def __init__(self, plant: Bicycle, horizon: int = 4, dt: float = 0.1):
        self.plant = plant
        self.horizon = horizon
        self.dt = dt
        self.warm_start = np.zeros((horizon, 2))
        self.last_solve_ms = 0.0
        self.last_success = True

    def _cost(self, flat_controls: Array, initial: Array) -> float:
        controls = flat_controls.reshape(self.horizon, 2)
        state = initial.copy()
        total = 0.0
        previous = controls[0]
        for control in controls:
            state = self.plant.step(state, control, self.dt)
            ref = reference_at(state[0])
            lateral = state[1] - ref[1]
            heading = wrap_angle(state[2] - ref[2])
            speed = state[3] - ref[3]
            total += (
                18 * lateral**2
                + 8 * heading**2
                + 0.7 * speed**2
                + 0.12 * control[0] ** 2
                + 0.4 * control[1] ** 2
            )
            total += 0.25 * np.sum((control - previous) ** 2)
            previous = control
        return float(total)

    def command(self, state: Array) -> Array:
        bounds = [
            (self.plant.params.min_accel_mps2, self.plant.params.max_accel_mps2),
            (-self.plant.params.max_steer_rad, self.plant.params.max_steer_rad),
        ] * self.horizon
        started = perf_counter_ns()
        result = minimize(
            self._cost,
            self.warm_start.ravel(),
            args=(state,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 12, "ftol": 2e-5},
        )
        self.last_solve_ms = (perf_counter_ns() - started) / 1e6
        self.last_success = bool(result.success)
        solution = result.x.reshape(self.horizon, 2) if np.all(np.isfinite(result.x)) else self.warm_start
        command = solution[0].copy()
        self.warm_start[:-1] = solution[1:]
        self.warm_start[-1] = solution[-1]
        return command


def run_episode(
    controller_name: str = "mpc",
    duration_s: float = 8.0,
    sample_time_s: float = 0.1,
    friction_scale: float = 1.0,
    delay_steps: int = 0,
) -> dict:
    plant = Bicycle()
    controllers = {
        "pure-pursuit": PurePursuit(plant),
        "lqr": LateralLQR(plant),
        "mpc": NonlinearMPC(plant, dt=sample_time_s),
    }
    controller = controllers[controller_name]
    time = np.arange(0, duration_s + sample_time_s / 2, sample_time_s)
    states = np.zeros((len(time), 4))
    controls = np.zeros((len(time), 2))
    solve_times = np.zeros(len(time))
    states[0] = [0, 0.8, 0, 3]
    queue = [np.zeros(2) for _ in range(delay_steps + 1)]
    for index in range(len(time) - 1):
        requested = controller.command(states[index])
        queue.append(requested)
        applied = queue.pop(0)
        controls[index] = applied
        states[index + 1] = plant.step(states[index], applied, sample_time_s, friction_scale)
        solve_times[index] = getattr(controller, "last_solve_ms", 0.0)
    refs = np.vstack([reference_at(x) for x in states[:, 0]])
    lateral = states[:, 1] - refs[:, 1]
    violations = int(
        np.count_nonzero(np.abs(lateral) > 1.5)
        + np.count_nonzero(np.abs(controls[:, 1]) > plant.params.max_steer_rad + 1e-9)
    )
    positive_solves = solve_times[solve_times > 0]
    return {
        "time": time,
        "states": states,
        "controls": controls,
        "references": refs,
        "lateral_rmse_m": float(np.sqrt(np.mean(lateral**2))),
        "constraint_violations": violations,
        "median_solve_time_ms": float(np.median(positive_solves)) if len(positive_solves) else 0.0,
        "p99_solve_time_ms": float(np.percentile(positive_solves, 99)) if len(positive_solves) else 0.0,
    }
