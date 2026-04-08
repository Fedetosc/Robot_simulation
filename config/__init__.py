"""
Configuration module - loads and validates simulation parameters.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class SimulationConfig:
    """Simulation timing and environment settings."""
    policy_path: str
    xml_path: str
    simulation_dt: float
    simulation_duration: float
    control_decimation: int


@dataclass
class ControlConfig:
    """PD controller gains and default joint positions."""
    kps: List[float]
    kds: List[float]
    default_angles: List[float]
    num_actions: int


@dataclass
class ObservationConfig:
    """Observation scaling and dimensions."""
    ang_vel_scale: float
    dof_pos_scale: float
    dof_vel_scale: float
    action_scale: float
    cmd_scale: List[float]
    num_obs: int
    cmd_init: List[float]


def load_config(config_path: Path | None = None) -> tuple[SimulationConfig, ControlConfig, ObservationConfig]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml. Defaults to same directory as this module.

    Returns:
        Tuple of (SimulationConfig, ControlConfig, ObservationConfig)
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    sim_cfg = SimulationConfig(
        policy_path=cfg["policy_path"],
        xml_path=cfg["xml_path"],
        simulation_dt=cfg["simulation_dt"],
        simulation_duration=cfg["simulation_duration"],
        control_decimation=cfg["control_decimation"],
    )

    ctrl_cfg = ControlConfig(
        kps=cfg["kps"],
        kds=cfg["kds"],
        default_angles=cfg["default_angles"],
        num_actions=cfg["num_actions"],
    )

    obs_cfg = ObservationConfig(
        ang_vel_scale=cfg["ang_vel_scale"],
        dof_pos_scale=cfg["dof_pos_scale"],
        dof_vel_scale=cfg["dof_vel_scale"],
        action_scale=cfg["action_scale"],
        cmd_scale=cfg["cmd_scale"],
        num_obs=cfg["num_obs"],
        cmd_init=cfg["cmd_init"],
    )

    return sim_cfg, ctrl_cfg, obs_cfg
