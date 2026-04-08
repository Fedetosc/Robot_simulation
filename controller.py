"""
PD Controller for joint position control.
"""

import numpy as np
from numpy.typing import NDArray


def pd_control(
    target_q: NDArray[np.floating],
    q: NDArray[np.floating],
    kp: NDArray[np.floating],
    target_dq: NDArray[np.floating],
    dq: NDArray[np.floating],
    kd: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Compute PD control torque for joint position control.

    Args:
        target_q: Target joint positions [rad] (n_joints,)
        q: Current joint positions [rad] (n_joints,)
        kp: Proportional gains (n_joints,)
        target_dq: Target joint velocities [rad/s] (n_joints,)
        dq: Current joint velocities [rad/s] (n_joints,)
        kd: Derivative gains (n_joints,)

    Returns:
        Torque commands [Nm] (n_joints,)
    """
    return (target_q - q) * kp + (target_dq - dq) * kd
