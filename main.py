import argparse
from simulation.runner import run_simulation


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Avvia la simulazione del robot")
    parser.add_argument(
        "--server",
        action="store_true",
        help="Avvia il server socket per ricevere comandi esterni"
    )
    args = parser.parse_args()

    run_simulation(is_server=args.server)