"""
Commander per G1 Simulation.
Uso:
  python commander.py forward          # avanti a 0.5 m/s
  python commander.py back             # indietro
  python commander.py left             # laterale sx
  python commander.py right            # laterale dx
  python commander.py rotate_left      # ruota sx
  python commander.py rotate_right     # ruota dx
  python commander.py stop             # stop
  python commander.py reset            # reset simulazione
  python commander.py state            # leggi stato robot
  python commander.py cmd 0.5 0.0 0.0  # comando custom [vx, vy, vyaw]
"""

import sys
import socket
import json


HOST = "localhost"
PORT = 9999


def send(msg: dict) -> dict:
    """Invia un comando JSON e ritorna la risposta."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(json.dumps(msg).encode())
        response = s.recv(4096).decode()
        return json.loads(response)


# Mappa comandi → messaggio JSON
COMMANDS = {
    "forward":       {"cmd": [ 0.5,  0.0,  0.0]},
    "back":          {"cmd": [-0.5,  0.0,  0.0]},
    "left":          {"cmd": [ 0.0,  0.3,  0.0]},
    "right":         {"cmd": [ 0.0, -0.3,  0.0]},
    "rotate_left":   {"cmd": [ 0.0,  0.0,  0.5]},
    "rotate_right":  {"cmd": [ 0.0,  0.0, -0.5]},
    "stop":          {"stop": True},
    "reset":         {"reset": True},
    "state":         {"get_state": True},
    "arms_up": {"arm_cmd": [
    0.0,  0.0,  0.0,          # waist yaw, roll, pitch
    0.5,  0.0,  0.0,  0.5,    # left shoulder + elbow
    0.0,  0.0,  0.0,          # left wrist
    0.0,  0.0,  0.0,          # left thumb
    0.0,  0.0,                # left middle
    0.0,  0.0,                # left index
    -0.5, 0.0,  0.0, -0.5,   # right shoulder + elbow
    0.0,  0.0,  0.0,          # right wrist
    0.0,  0.0,  0.0,          # right thumb
    0.0,  0.0,                # right index
    0.0,  0.0,                # right middle
]},
"arms_zero": {"arm_cmd": [0.0] * 31},
}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "cmd":
        if len(sys.argv) < 5:
            print("Uso: python commander.py cmd <vx> <vy> <vyaw>")
            sys.exit(1)
        msg = {"cmd": [float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])]}

    elif command in COMMANDS:
        msg = COMMANDS[command]

    else:
        print(f"Comando sconosciuto: {command}")
        print(f"Comandi disponibili: {list(COMMANDS.keys())} + cmd")
        sys.exit(1)

    try:
        response = send(msg)
        print(json.dumps(response, indent=2))
    except ConnectionRefusedError:
        print("ERRORE: Simulazione non in esecuzione. Avvia prima simulation.py")