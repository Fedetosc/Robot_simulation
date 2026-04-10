"""
Runner module - entry point for running the simulation.
"""

import logging

from config import load_config
from simulation.simulation import Simulation

# Configura logger root
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


def run_simulation(is_server: bool = False) -> None:
    """
    Funzione entry point per avviare la simulazione.

    Args:
        is_server: Se True, avvia il server socket per ricevere comandi esterni
                   e imposta il comando iniziale a zero (nessun movimento automatico)
    """
    logger.info("=" * 60)
    logger.info("=== Avvio Simulazione Robot ===")
    logger.info("=" * 60)

    # Caricamento configurazioni dai file YAML
    logger.info("Caricamento configurazioni...")
    sim_cfg, ctrl_cfg, obs_cfg = load_config()

    logger.info(f"Simulation config: dt={sim_cfg.simulation_dt}, decimation={sim_cfg.control_decimation}")
    logger.info(f"Control config: num_actions={ctrl_cfg.num_actions}")
    logger.info(f"Observation config: cmd_init={obs_cfg.cmd_init}")

    # Se server, ignora il cammino automatico iniziale
    if is_server:
        logger.info("Modalità SERVER: comando iniziale azzerato (attesa comandi esterni)")
        obs_cfg.cmd_init = [0.0, 0.0, 0.0]
    else:
        logger.info(f"Modalità STANDALONE: comando iniziale = {obs_cfg.cmd_init}")

    # Creazione istanza della simulazione
    logger.info("Creazione istanza Simulation...")
    sim = Simulation(sim_cfg, ctrl_cfg, obs_cfg, start_server=is_server)

    # Avvio del loop principale
    logger.info("Avvio simulazione...\n")
    sim.run()

    logger.info("=== Simulazione terminata ===\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Avvia la simulazione del robot quadrupede")
    parser.add_argument(
        "--server",
        action="store_true",
        help="Avvia il server socket per ricevere comandi esterni"
    )
    args = parser.parse_args()

    run_simulation(is_server=args.server)
