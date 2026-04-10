"""
Factory module - creates the appropriate simulation class based on XML model.
"""

import logging
from pathlib import Path

from config import ControlConfig, ObservationConfig, SimulationConfig
from .simulation_g1 import SimulationG1
from .simulation_g1_with_hand import SimulationG1WithHand

logger = logging.getLogger(__name__)


def get_simulation_class(xml_path: str) -> type:
    """
    Selects the appropriate simulation class based on the XML model name.

    Args:
        xml_path: Path to the MJCF XML model file

    Returns:
        Simulation class (SimulationG1 or SimulationG1WithHand)

    Rules:
        - scene.xml or *12dof* -> SimulationG1
        - *with_hand* or *29dof* (con mani) -> SimulationG1WithHand
    """
    xml_name = Path(xml_path).name.lower()

    # Contiene "hand" -> usa versione con braccia/mani
    if "hand" in xml_name:
        logger.info(f"Rilevato modello con mani: {xml_name} -> SimulationG1WithHand")
        return SimulationG1WithHand

    # Altrimenti usa versione base (solo gambe)
    logger.info(f"Rilevato modello base: {xml_name} -> SimulationG1")
    return SimulationG1


def create_simulation(
    sim_cfg: SimulationConfig,
    ctrl_cfg: ControlConfig,
    obs_cfg: ObservationConfig,
    start_server: bool = False
):
    """
    Factory function that creates and returns the appropriate simulation instance.

    Args:
        sim_cfg: Simulation configuration
        ctrl_cfg: Control configuration
        obs_cfg: Observation configuration
        start_server: Whether to start the command server

    Returns:
        Simulation instance (SimulationG1 or SimulationG1WithHand)
    """
    sim_class = get_simulation_class(sim_cfg.xml_path)
    return sim_class(sim_cfg, ctrl_cfg, obs_cfg, start_server)
