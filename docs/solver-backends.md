# Solver backend contract

The optimization state is `[x, y, yaw, speed]`; input is `[acceleration, steering]`. Each stage penalizes cross-track, heading, speed, input, and input-rate errors. Acceleration and steering use hard bounds.

The default SciPy backend keeps CI installation small. An acados backend must implement `command(state) -> [acceleration, steering]`, preserve these bounds, expose solve time and status, and pass the same benchmark before its results are published.

