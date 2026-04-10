"""
Simulation package - quadruped robot simulation with RL control.

Supports multiple G1 robot configurations:
- SimulationG1: 12-DOF (legs only)
- SimulationG1WithHand: 29-DOF (legs + arms + hands)

The appropriate class is automatically selected based on the XML model name.
"""

from .simulation_g1 import SimulationG1
from .simulation_g1_with_hand import SimulationG1WithHand
from .factory import create_simulation
from .runner import run_simulation

__all__ = [
    "SimulationG1",
    "SimulationG1WithHand",
    "create_simulation",
    "run_simulation",
]
