from config.logger_config import setup_logger
setup_logger()  # Configura subito il logger globale

import logging
logger = logging.getLogger(__name__)  # Logger per il modulo

import sys 
from simulation import run_simulation

if __name__ == "__main__":
    # Controlla se l'utente ha passato l'argomento 'server'
    use_server = False
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        use_server = True
        logger.info("--- MODALITÀ SERVER ATTIVA: In attesa di commander.py ---")
    else:
        logger.info("--- MODALITÀ AUTOMATICA: Uso cmd_init dallo YAML ---")

    # Passiamo l'informazione alla funzione run_simulation
    run_simulation(is_server=use_server)