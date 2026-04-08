import sys
from simulation import run_simulation

if __name__ == "__main__":
    # Controlla se l'utente ha passato l'argomento 'server'
    use_server = False
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        use_server = True
        print("--- MODALITÀ SERVER ATTIVA: In attesa di commander.py ---")
    else:
        print("--- MODALITÀ AUTOMATICA: Uso cmd_init dallo YAML ---")

    # Passiamo l'informazione alla funzione run_simulation
    run_simulation(is_server=use_server)