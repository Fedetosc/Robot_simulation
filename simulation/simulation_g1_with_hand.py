"""
Simulation for G1 29-DOF robot with hands.
Extended version that controls both legs and arms/hands.
"""

import time
import json
import socket
import threading
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from config import ControlConfig, ObservationConfig, SimulationConfig
from controller import pd_control
from observer import build_observation, compute_gait_phase, get_gravity_orientation

import logging
logger = logging.getLogger(__name__)


class SimulationG1WithHand:
    """
    Simulazione per robot G1 con 29 DOF (gambe + braccia + mani).
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

        logger.info("Inizializzazione simulazione G1 With Hand (29 DOF)...")

        # Conversione parametri - guadagni PD e angoli di default
        self.kps = np.array(ctrl_cfg.kps, dtype=np.float32)
        self.kds = np.array(ctrl_cfg.kds, dtype=np.float32)
        self.default_angles = np.array(ctrl_cfg.default_angles, dtype=np.float32)
        self.cmd_scale = np.array(obs_cfg.cmd_scale, dtype=np.float32)
        self.cmd = np.array(obs_cfg.cmd_init, dtype=np.float32)

        logger.debug(f"KPS: {self.kps}")
        logger.debug(f"KDS: {self.kds}")
        logger.debug(f"Default angles: {self.default_angles}")
        logger.info(f"Comando iniziale: {self.cmd}")

        # Variabili di stato
        self.action = np.zeros(ctrl_cfg.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.counter = 0
        self.stopped = False
        self.frozen_qpos = None
        self.frozen_qvel = None

        # Caricamento modello MuJoCo
        logger.info(f"Caricamento modello MuJoCo da: {sim_cfg.xml_path}")
        self.model = mujoco.MjModel.from_xml_path(sim_cfg.xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = sim_cfg.simulation_dt
        logger.info(f"Modello caricato - Timestep: {self.model.opt.timestep}s")

        # DOF totali dal modello
        self.total_dofs = self.model.nu
        logger.info(f"Numero totale attuatori: {self.total_dofs}")

        # Nomi attuatori per identificare gambe vs braccia
        self.actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.total_dofs)
        ]
        logger.info(f"Attuatori: {self.actuator_names}")

        # Identifica indici delle gambe (contengono hip, knee, ankle)
        leg_keywords = ["hip", "knee", "ankle"]
        self.leg_indices = np.array([
            i for i, name in enumerate(self.actuator_names)
            if any(k in name.lower() for k in leg_keywords)
        ])
        self.other_indices = np.setdiff1d(np.arange(self.total_dofs), self.leg_indices)
        logger.info(f"Leg indices ({len(self.leg_indices)}): {self.leg_indices}")
        logger.info(f"Other indices ({len(self.other_indices)}): {self.other_indices}")

        # Target per tutti i DOF
        self.target_dof_pos_full = np.zeros(self.total_dofs, dtype=np.float32)
        self.target_dof_pos_full[self.leg_indices] = self.default_angles
        self.arm_target = np.zeros(len(self.other_indices), dtype=np.float32)

        # Caricamento Policy RL
        logger.info(f"Caricamento policy RL da: {sim_cfg.policy_path}")
        self.policy = torch.jit.load(Path(sim_cfg.policy_path))
        self.policy.eval()
        logger.info("Policy RL caricata e pronta")

        # Avvio Server Socket se richiesto
        if start_server:
            self.host = "localhost"
            self.port = 9999
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()
            logger.info(f"Server in ascolto su {self.host}:{self.port}")

    def _run_server(self) -> None:
        """Thread per ricevere comandi JSON dal commander.py."""
        logger.info("Avvio thread server socket...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen()
            logger.info(f"Socket bindato su {self.host}:{self.port}")

            while True:
                conn, addr = s.accept()
                logger.info(f"Connessione ricevuta da {addr}")

                with conn:
                    data = conn.recv(1024).decode()
                    if not data:
                        logger.info("Nessun dato ricevuto")
                        continue

                    msg = json.loads(data)
                    logger.info(f"Messaggio ricevuto: {msg}")
                    response = {"status": "ok"}

                    if "cmd" in msg:
                        old_cmd = self.cmd.copy()
                        self.stopped = False
                        self.cmd = np.array(msg["cmd"], dtype=np.float32)
                        response["new_cmd"] = self.cmd.tolist()
                        logger.info(f"Nuovo comando: {old_cmd} -> {self.cmd}")

                    if "stop" in msg:
                        self.stopped = True
                        self.frozen_qpos = self.data.qpos.copy()
                        self.frozen_qvel = self.data.qvel.copy()
                        self.cmd = np.zeros(3, dtype=np.float32)
                        response["status"] = "stopped"
                        logger.info("STOP - Robot congelato")

                    if "reset" in msg:
                        mujoco.mj_resetData(self.model, self.data)
                        mujoco.mj_forward(self.model, self.data)
                        self.counter = 0
                        self.stopped = False
                        self.cmd = np.zeros(3, dtype=np.float32)
                        self.target_dof_pos_full[self.leg_indices] = self.default_angles
                        logger.info("RESET - Simulazione resettata")

                    if "get_state" in msg:
                        response["pos"] = self.data.qpos[:3].tolist()
                        logger.info(f"Stato inviato - Posizione: {response['pos']}")

                    if "arm_cmd" in msg:
                        # Comando per braccia/mani
                        arm_target = np.array(msg["arm_cmd"], dtype=np.float32)
                        self.arm_target = arm_target
                        response["arm_cmd"] = arm_target.tolist()
                        logger.info(f"Nuovo comando braccia: {arm_target}")

                    conn.sendall(json.dumps(response).encode())

    def _update_policy(self) -> None:
        """Inference della rete neurale per le gambe."""
        qj = self.data.qpos[7:7+self.ctrl_cfg.num_actions]
        dqj = self.data.qvel[6:6+self.ctrl_cfg.num_actions]
        quat = self.data.qpos[3:7]
        omega = self.data.qvel[3:6]

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

        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze().astype(np.float32)
        self.target_dof_pos = self.action * self.obs_cfg.action_scale + self.default_angles
        self.target_dof_pos_full[self.leg_indices] = self.target_dof_pos

    def step(self) -> None:
        """Passo di simulazione."""
        if self.stopped:
            self.data.qvel[:] = 0
            self.data.ctrl[:] = 0
            self.data.qpos[:] = self.frozen_qpos
            mujoco.mj_forward(self.model, self.data)
            return

        current_q = self.data.qpos[7:7+self.total_dofs]
        current_dq = self.data.qvel[6:6+self.total_dofs]

        kp_full = np.zeros(self.total_dofs, dtype=np.float32)
        kd_full = np.zeros(self.total_dofs, dtype=np.float32)

        if np.linalg.norm(self.cmd) > 0.05:
            kp_full[self.leg_indices] = self.kps
            kd_full[self.leg_indices] = self.kds
        else:
            kp_full[self.leg_indices] = self.kps
            kd_full[self.leg_indices] = self.kds

        # Braccia/mani: guadagni fissi
        kp_full[self.other_indices] = 80.0
        kd_full[self.other_indices] = 5.0
        self.target_dof_pos_full[self.other_indices] = self.arm_target

        self.data.ctrl[:self.total_dofs] = pd_control(
            target_q=self.target_dof_pos_full,
            q=current_q,
            kp=kp_full,
            target_dq=np.zeros(self.total_dofs),
            dq=current_dq,
            kd=kd_full,
        )

        mujoco.mj_step(self.model, self.data)
        self.counter += 1

        # Update policy ogni control_decimation step
        if self.counter % self.sim_cfg.control_decimation == 0:
            if np.linalg.norm(self.cmd) > 0.05:
                self._update_policy()
            else:
                # Reset stato quando fermo
                self.data.qvel[:] = 0
                self.data.qacc[:] = 0
                self.data.qfrc_applied[:] = 0
                self.data.qpos[3:7] = [1, 0, 0, 0]
                self.data.qpos[7:7+self.ctrl_cfg.num_actions] = self.default_angles
                self.target_dof_pos = self.default_angles.copy()
                self.target_dof_pos_full[self.leg_indices] = self.default_angles
                mujoco.mj_forward(self.model, self.data)

    def run(self) -> None:
        """Loop principale con visualizzatore."""
        logger.info("Avvio loop principale...")

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance = 5.0
            viewer.cam.elevation = -15
            viewer.cam.azimuth = 90

            logger.info(f"Visualizzatore avviato - Camera: distance={viewer.cam.distance}, "
                       f"elevation={viewer.cam.elevation}, azimuth={viewer.cam.azimuth}")

            while viewer.is_running():
                step_start = time.time()
                self.step()
                viewer.sync()

                elapsed = time.time() - step_start
                if self.model.opt.timestep > elapsed:
                    time.sleep(self.model.opt.timestep - elapsed)

            logger.info("Simulazione terminata")
