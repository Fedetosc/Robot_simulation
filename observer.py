"""
Observation builder for RL policy input.
"""

import numpy as np
from numpy.typing import NDArray


def get_gravity_orientation(quaternion: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Convert robot orientation quaternion to gravity vector in robot frame.

    The gravity vector indicates robot tilt - used by RL policy for balance.

    Args:
        quaternion: Orientation quaternion [qw, qx, qy, qz]

    Returns:
        Gravity vector [gx, gy, gz] in robot frame
    """
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]

    gravity = np.zeros(3, dtype=np.float32)
    gravity[0] = 2 * (-qz * qx + qw * qy)
    gravity[1] = -2 * (qz * qy + qw * qx)
    gravity[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity


def build_observation(
    omega: NDArray[np.floating],
    gravity: NDArray[np.floating],
    cmd: NDArray[np.floating],
    qj: NDArray[np.floating],
    dqj: NDArray[np.floating],
    action: NDArray[np.floating],
    phase: float,
    ang_vel_scale: float,
    dof_pos_scale: float,
    dof_vel_scale: float,
    cmd_scale: NDArray[np.floating],
    default_angles: NDArray[np.floating],
    num_actions: int,
) -> NDArray[np.floating]:
    """
    Build observation vector for RL policy.

    Observation layout (47 dims):
        [0:3]   - Angular velocity (IMU)
        [3:6]   - Gravity orientation
        [6:9]   - Command velocity (scaled)
        [9:21]  - Joint positions (scaled)
        [21:33] - Joint velocities (scaled)
        [33:45] - Previous action
        [45:47] - Phase (sin, cos)

    Args:
        omega: Angular velocity from IMU [rad/s] (3,)
        gravity: Gravity vector from quaternion (3,)
        cmd: Command velocity [vx, vy, vyaw] (3,)
        qj: Joint positions [rad] (12,)
        dqj: Joint velocities [rad/s] (12,)
        action: Previous policy action (12,)
        phase: Gait phase [0, 1]
        ang_vel_scale: Scaling factor for angular velocity
        dof_pos_scale: Scaling factor for joint positions
        dof_vel_scale: Scaling factor for joint velocities
        cmd_scale: Scaling factors for command velocity (3,)
        default_angles: Default joint positions (12,)
        num_actions: Number of controlled joints (12)

    Returns:
        Observation vector (47,)
    """
    obs = np.zeros(47, dtype=np.float32)

    # Scale inputs
    omega_scaled = omega * ang_vel_scale
    qj_scaled = (qj - default_angles) * dof_pos_scale
    dqj_scaled = dqj * dof_vel_scale

    # Phase encoding
    sin_phase = np.sin(2 * np.pi * phase)
    cos_phase = np.cos(2 * np.pi * phase)

    # Build observation vector
    obs[:3] = omega_scaled
    obs[3:6] = gravity
    obs[6:9] = cmd * cmd_scale
    obs[9:9+num_actions] = qj_scaled
    obs[9+num_actions:9+2*num_actions] = dqj_scaled
    obs[9+2*num_actions:9+3*num_actions] = action
    obs[9+3*num_actions:9+3*num_actions+2] = [sin_phase, cos_phase]

    return obs


def compute_gait_phase(counter: int, simulation_dt: float, period: float = 0.8) -> float:
    """
    Compute gait phase from simulation counter.

    Args:
        counter: Simulation step counter
        simulation_dt: Simulation timestep [s]
        period: Gait period [s] (default 0.8s)

    Returns:
        Phase value [0, 1]
    """
    elapsed_time = counter * simulation_dt
    return (elapsed_time % period) / period
