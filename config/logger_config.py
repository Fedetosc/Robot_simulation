# logger_config.py
import logging
import sys

def setup_logger(level=logging.INFO):
    """
    Configura il logger globale per tutta l'applicazione.
    """
    # Logger root
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Evita di aggiungere più handler se già presente
    if not root_logger.handlers:
        # StreamHandler per il terminale
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Formattazione dei messaggi
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)