import numpy as np

from vehicle_mpc.core import Bicycle, NonlinearMPC, reference_at, run_episode


def test_straight_motion_and_dimensions():
    plant = Bicycle()
    derivative = plant.derivative(np.array([0, 0, 0, 5.0]), np.zeros(2))
    assert derivative.shape == (4,)
    assert np.allclose(derivative, [5, 0, 0, 0])


def test_mpc_obeys_input_bounds():
    plant = Bicycle()
    control = NonlinearMPC(plant, horizon=3).command(np.array([0, 2, 0.3, 2]))
    assert plant.params.min_accel_mps2 <= control[0] <= plant.params.max_accel_mps2
    assert abs(control[1]) <= plant.params.max_steer_rad


def test_reference_is_smooth():
    left, center, right = reference_at(0.99), reference_at(1.0), reference_at(1.01)
    assert np.linalg.norm(right - center) < 0.1
    assert np.linalg.norm(center - left) < 0.1


def test_nominal_mpc_tracks_inside_lane():
    result = run_episode("mpc", duration_s=4.0)
    assert result["constraint_violations"] == 0
    assert result["lateral_rmse_m"] < 0.6
