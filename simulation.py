import time
import socket
import threading
import json
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from config import load_config, ControlConfig, ObservationConfig, SimulationConfig
from controller import pd_control
from observer import build_observation, compute_gait_phase, get_gravity_orientation

import logging
logger = logging.getLogger(__name__)


class Simulation:
    """
    Classe principale per la simulazione del robot quadrupede.
    Gestisce il ciclo di simulazione MuJoCo, il controllo RL e opzionalmente
    un server socket per ricevere comandi esterni.
    """
    def __init__(
        self,
        sim_cfg: SimulationConfig,
        ctrl_cfg: ControlConfig,
        obs_cfg: ObservationConfig,
        start_server: bool = False
    ):
        self.sim_cfg = sim_cfg
        self.ctrl_cfg = ctrl_cfg
        self.obs_cfg = obs_cfg

        logger.info("Inizializzazione simulazione...")

        # Conversione parametri - guadagni PD e angoli di default
        self.kps = np.array(ctrl_cfg.kps, dtype=np.float32)  # Guadagni proporzionali (posizione)
        self.kds = np.array(ctrl_cfg.kds, dtype=np.float32)  # Guadagni derivativi (velocità)
        self.default_angles = np.array(ctrl_cfg.default_angles, dtype=np.float32)  # Angoli di riposo delle giunture
        self.cmd_scale = np.array(obs_cfg.cmd_scale, dtype=np.float32)  # Fattori di scala per i comandi
        self.cmd = np.array(obs_cfg.cmd_init, dtype=np.float32)  # Comando corrente [vx, vy, yaw]

        logger.debug(f"KPS: {self.kps}")
        logger.debug(f"KDS: {self.kds}")
        logger.debug(f"Default angles: {self.default_angles}")
        logger.info(f"Comando iniziale: {self.cmd}")

        # Variabili di stato
        self.action = np.zeros(ctrl_cfg.num_actions, dtype=np.float32)  # Output della policy RL
        self.target_dof_pos = self.default_angles.copy()  # Posizioni target delle giunture
        self.counter = 0  # Contatore passi di simulazione

        # Caricamento modello MuJoCo dal file XML
        logger.info(f"Caricamento modello MuJoCo da: {sim_cfg.xml_path}")
        self.model = mujoco.MjModel.from_xml_path(sim_cfg.xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = sim_cfg.simulation_dt  # Intervallo di simulazione (es. 0.002s = 500Hz)
        logger.info(f"Modello caricato - Timestep: {self.model.opt.timestep}s")

        # Caricamento Policy RL (rete neurale pre-addestrata)
        logger.info(f"Caricamento policy RL da: {sim_cfg.policy_path}")
        self.policy = torch.jit.load(Path(sim_cfg.policy_path))
        self.policy.eval()  # Modalità inference (no gradienti)
        logger.info("Policy RL caricata e pronta")

        # Avvio Server Socket se richiesto (per ricevere comandi da commander.py)
        if start_server:
            self.host = "localhost"
            self.port = 9999
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()
            logger.info(f"Server in ascolto su {self.host}:{self.port}")

    def _run_server(self):
        """
        Thread separato per ricevere comandi JSON dal commander.py.
        Gestisce: cambio comando, stop, reset simulazione, richiesta stato.
        """
        logger.info("[SERVER] Avvio thread server socket...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen()
            logger.info(f"[SERVER] Socket bindato su {self.host}:{self.port}, in ascolto...")

            while True:
                conn, addr = s.accept()
                logger.info(f"[SERVER] Connessione ricevuta da {addr}")

                with conn:
                    data = conn.recv(1024).decode()
                    if not data:
                        logger.info("[SERVER] Nessun dato ricevuto, chiudo connessione")
                        continue

                    msg = json.loads(data)
                    logger.info(f"[SERVER] Messaggio ricevuto: {msg}")
                    response = {"status": "ok"}

                    if "cmd" in msg:
                        old_cmd = self.cmd.copy()
                        self.cmd = np.array(msg["cmd"], dtype=np.float32)
                        response["new_cmd"] = self.cmd.tolist()
                        logger.info(f"[SERVER] Nuovo comando: {old_cmd} -> {self.cmd}")

                    if "stop" in msg:
                        self.cmd = np.zeros(3, dtype=np.float32)
                        logger.info("[SERVER] STOP - Comando azzerato")

                    if "reset" in msg:
                        mujoco.mj_resetData(self.model, self.data)
                        self.counter = 0
                        logger.info("[SERVER] RESET - Dati simulazione resettati")

                    if "get_state" in msg:
                        response["pos"] = self.data.qpos[:3].tolist()
                        logger.info(f"[SERVER] Stato inviato - Posizione: {response['pos']}")

                    conn.sendall(json.dumps(response).encode())
                    logger.info(f"[SERVER] Risposta inviata: {response}")

    def _update_policy(self) -> None:
        """
        Esegue l'inference della rete neurale per calcolare le nuove azioni.
        Chiamata ogni N passi di simulazione (control_decimation).
        """
        # Estrazione stato corrente dal modello MuJoCo
        qj = self.data.qpos[7:7+self.ctrl_cfg.num_actions]  # Posizioni angolari delle giunture
        dqj = self.data.qvel[6:6+self.ctrl_cfg.num_actions]  # Velocità angolari delle giunture
        quat = self.data.qpos[3:7]  # Orientamento del corpo (quaternioni)
        omega = self.data.qvel[3:6]  # Velocità angolare del corpo

        # print(f"[POLICY] === Inference RL ===")
        # print(f"[POLICY]   qj (pos giunture): {qj}")
        # print(f"[POLICY]   dqj (vel giunture): {dqj}")
        # print(f"[POLICY]   quat (orientamento): {quat}")
        # print(f"[POLICY]   omega (vel angolare): {omega}")
        # print(f"[POLICY]   cmd (comando): {self.cmd}")

        # Costruzione del vettore di osservazione per la policy
        obs = build_observation(
            omega=omega, gravity=get_gravity_orientation(quat),
            cmd=self.cmd, qj=qj, dqj=dqj, action=self.action,
            phase=compute_gait_phase(self.counter, self.sim_cfg.simulation_dt),
            ang_vel_scale=self.obs_cfg.ang_vel_scale,
            dof_pos_scale=self.obs_cfg.dof_pos_scale,
            dof_vel_scale=self.obs_cfg.dof_vel_scale,
            cmd_scale=self.cmd_scale,
            default_angles=self.default_angles,
            num_actions=self.ctrl_cfg.num_actions,
        )

        # print(f"[POLICY]   Osservazione costruita (shape: {obs.shape})")

        # Inference della rete neurale
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)  # Aggiungo batch dimension
        self.action = self.policy(obs_tensor).detach().numpy().squeeze().astype(np.float32)

        # Calcolo posizioni target dalle azioni della policy
        self.target_dof_pos = self.action * self.obs_cfg.action_scale + self.default_angles

        # print(f"[POLICY]   Output policy: {self.action}")
        # print(f"[POLICY]   Target dof pos: {self.target_dof_pos}")
        # print(f"[POLICY] ===================")

    def step(self) -> None:
        current_q = self.data.qpos[7:7+self.ctrl_cfg.num_actions]
        current_dq = self.data.qvel[6:6+self.ctrl_cfg.num_actions]

        # Scegli guadagni PRIMA del PD
        if np.linalg.norm(self.cmd) > 0.05:
            kps = self.kps.copy()
            kds = self.kds.copy()
        else:
            kps = self.kps.copy()
            kds = self.kds.copy()
            kps[4]  = 200.0
            kps[5]  = 200.0
            kps[10] = 200.0
            kps[11] = 200.0
            kds[4]  = 20.0
            kds[5]  = 20.0
            kds[10] = 20.0
            kds[11] = 20.0
            kps[3]  = 300.0
            kps[9]  = 300.0
            kds[3]  = 15.0
            kds[9]  = 15.0

        self.data.ctrl[:self.ctrl_cfg.num_actions] = pd_control(
            target_q=self.target_dof_pos,
            q=current_q,
            kp=kps,
            target_dq=np.zeros(self.ctrl_cfg.num_actions),
            dq=current_dq,
            kd=kds,
        )

        mujoco.mj_step(self.model, self.data)
        self.counter += 1

        # if self.counter % 50 == 0 and np.linalg.norm(self.cmd) <= 0.05:
            # print(f"qpos base: {self.data.qpos[:3]}")
            # print(f"quat: {self.data.qpos[3:7]}")
            # print(f"joints: {self.data.qpos[7:19]}")
            # print(f"target: {self.target_dof_pos}")
            # print(f"errore: {self.target_dof_pos - self.data.qpos[7:19]}")

        if self.counter % self.sim_cfg.control_decimation == 0:
            if np.linalg.norm(self.cmd) > 0.05:
                self._update_policy()
            else:
                self.data.qvel[:] = 0
                self.data.qacc[:] = 0
                self.data.qfrc_applied[:] = 0
                self.data.qpos[3:7] = [1, 0, 0, 0]
                self.data.qpos[7:7+self.ctrl_cfg.num_actions] = self.default_angles
                self.target_dof_pos = self.default_angles.copy()

    def run(self) -> None:
        """
        Loop principale di simulazione con visualizzatore MuJoCo.
        Esegue il ciclo di simulazione sincronizzandosi con il tempo reale.
        """
        logger.info("[RUN] Avvio loop principale di simulazione...")

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Configurazione iniziale della camera
            viewer.cam.distance = 5.0  # Distanza dal centro
            viewer.cam.elevation = -15  # Angolo verticale (gradi)
            viewer.cam.azimuth = 90  # Angolo orizzontale (gradi)

            logger.info(f"[RUN] Visualizzatore avviato - Camera: distance={viewer.cam.distance}, elevation={viewer.cam.elevation}, azimuth={viewer.cam.azimuth}")
            logger.info("[RUN] Entro nel ciclo di simulazione...")

            step_count = 0
            while viewer.is_running():
                step_start = time.time()

                self.step()  # Esegue un passo di simulazione
                viewer.sync()  # Sincronizza il visualizzatore

                step_count += 1
                # if step_count % 100 == 0:
                #     logger.info(f"[RUN] ... {step_count} passi eseguiti ...")

                # Sincronizzazione tempo reale - aspetta se il passo è stato troppo veloce
                elapsed = time.time() - step_start
                if self.model.opt.timestep > elapsed:
                    time.sleep(self.model.opt.timestep - elapsed)

            logger.info(f"[RUN] Simulazione terminata dopo {step_count} passi totali")


def run_simulation(is_server: bool = False):
    """
    Funzione entry point per avviare la simulazione.

    Args:
        is_server: Se True, avvia il server socket per ricevere comandi esterni
                   e imposta il comando iniziale a zero (nessun movimento automatico)
    """
    logger.info("\n" + "="*60)
    logger.info("[MAIN] === Avvio Simulazione Robot ===")
    logger.info("="*60)

    # Caricamento configurazioni dai file YAML
    logger.info("[MAIN] Caricamento configurazioni...")
    sim_cfg, ctrl_cfg, obs_cfg = load_config()

    logger.info(f"[MAIN]   Simulation config: dt={sim_cfg.simulation_dt}, decimation={sim_cfg.control_decimation}")
    logger.info(f"[MAIN]   Control config: num_actions={ctrl_cfg.num_actions}")
    logger.info(f"[MAIN]   Observation config: cmd_init={obs_cfg.cmd_init}")

    # Se server, ignora il cammino automatico iniziale
    if is_server:
        logger.info("[MAIN] Modalità SERVER: comando iniziale azzerato (attesa comandi esterni)")
        obs_cfg.cmd_init = [0.0, 0.0, 0.0]
    else:
        logger.info("[MAIN] Modalità STANDALONE: comando iniziale =", obs_cfg.cmd_init)

    # Creazione istanza della simulazione
    logger.info("[MAIN] Creazione istanza Simulation...")
    sim = Simulation(sim_cfg, ctrl_cfg, obs_cfg, start_server=is_server)

    # Avvio del loop principale
    logger.info("[MAIN] Avvio simulazione...\n")
    sim.run()

    logger.info("[MAIN] === Simulazione terminata ===\n")