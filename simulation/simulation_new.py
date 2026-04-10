"""
Simulation module - main simulation class for G1 robot.

Architettura di controllo:
  500Hz  →  PD Control  →  coppie ai motori (tutti i DOF)
   50Hz  →  Policy RL   →  target gambe (12 DOF)

DOF layout (43 totali):
  [0-11]  GAMBE     → policy RL (motion.pt)
  [12-14] VITA      → PD pose fissa
  [15-21] BRACCIO SX → PD comandabile via socket
  [22-28] MANO SX   → PD comandabile via socket
  [29-35] BRACCIO DX → PD comandabile via socket
  [36-42] MANO DX   → PD comandabile via socket
"""

import time
import logging
import socket
import threading
import json
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from config import ControlConfig, ObservationConfig, SimulationConfig
from controller import pd_control
from observer import build_observation, compute_gait_phase, get_gravity_orientation

logger = logging.getLogger(__name__)


class Simulation:

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

        # ================================================================
        # PARAMETRI GENERALI
        # ================================================================
        self.cmd_scale = np.array(obs_cfg.cmd_scale, dtype=np.float32)
        self.cmd = np.array(obs_cfg.cmd_init, dtype=np.float32)
        logger.info(f"Comando iniziale: {self.cmd}")

        # ================================================================
        # CARICAMENTO MODELLO MUJOCO
        # ================================================================
        logger.info(f"Caricamento modello MuJoCo da: {sim_cfg.xml_path}")
        self.model = mujoco.MjModel.from_xml_path(sim_cfg.xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = sim_cfg.simulation_dt
        mujoco.mj_forward(self.model, self.data)  # calcola stato iniziale
        logger.info(f"Modello caricato - Timestep: {self.model.opt.timestep}s")

        
        # DOF totali dal modello
        self.total_dofs = self.model.nu
        self.actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.total_dofs)
        ]
        logger.info(f"Attuatori ({self.total_dofs}): {self.actuator_names}")

        # ================================================================
        # GAMBE — controllate dalla policy RL (12 DOF)
        # ================================================================
        leg_keywords = ["hip", "knee", "ankle"]
        self.leg_indices = np.array([
            i for i, name in enumerate(self.actuator_names)
            if any(k in name.lower() for k in leg_keywords)
        ])
        logger.info(f"Leg indices ({len(self.leg_indices)}): {self.leg_indices}")

        # Guadagni PD gambe (dal config YAML)
        self.leg_kps = np.array(ctrl_cfg.kps, dtype=np.float32)
        self.leg_kds = np.array(ctrl_cfg.kds, dtype=np.float32)

        # Pose default gambe (dal config YAML)
        self.leg_default_angles = np.array(ctrl_cfg.default_angles, dtype=np.float32)
        logger.debug(f"Leg KPS: {self.leg_kps}")
        logger.debug(f"Leg KDS: {self.leg_kds}")
        logger.debug(f"Leg default angles: {self.leg_default_angles}")

        # Stato policy RL
        self.action = np.zeros(ctrl_cfg.num_actions, dtype=np.float32)
        self.leg_target = self.leg_default_angles.copy()

        # ================================================================
        # VITA + BRACCIA + MANI — controllate via PD (31 DOF)
        # ================================================================
        self.other_indices = np.setdiff1d(np.arange(self.total_dofs), self.leg_indices)
        logger.info(f"Other indices ({len(self.other_indices)}): {self.other_indices}")

        # Guadagni PD per categoria (distale = guadagni più bassi)
        self.other_kp = np.zeros(self.total_dofs, dtype=np.float32)
        self.other_kd = np.zeros(self.total_dofs, dtype=np.float32)
        for idx in self.other_indices:
            name = self.actuator_names[idx]
            if "hand" in name:
                self.other_kp[idx] = 5.0
                self.other_kd[idx] = 0.5
            elif "wrist" in name:
                self.other_kp[idx] = 30.0
                self.other_kd[idx] = 3.0
            elif "elbow" in name:
                self.other_kp[idx] = 40.0
                self.other_kd[idx] = 5.0
            elif "shoulder" in name:
                self.other_kp[idx] = 60.0
                self.other_kd[idx] = 5.0
            elif "waist" in name:
                self.other_kp[idx] = 80.0
                self.other_kd[idx] = 8.0

        # Pose default braccia/mani/vita (letta dall'XML)
        qpos_default = self.data.qpos[7:7+self.total_dofs]
        self.arm_default = qpos_default[self.other_indices].copy().astype(np.float32)
        self.arm_target = self.arm_default.copy()
        logger.info(f"Arm target iniziale (da XML): {self.arm_target}")

        # ================================================================
        # TARGET UNIFICATO (tutti i DOF)
        # ================================================================
        self.target_dof_pos_full = np.zeros(self.total_dofs, dtype=np.float32)
        self.target_dof_pos_full[self.leg_indices] = self.leg_default_angles
        self.target_dof_pos_full[self.other_indices] = self.arm_default

        # ================================================================
        # POLICY RL
        # ================================================================
        logger.info(f"Caricamento policy RL da: {sim_cfg.policy_path}")
        self.policy = torch.jit.load(Path(sim_cfg.policy_path))
        self.policy.eval()
        logger.info("Policy RL caricata e pronta")

        self._log_initial_state()

        # ================================================================
        # SERVER SOCKET (opzionale)
        # ================================================================
        self.counter = 0
        if start_server:
            self.host = "localhost"
            self.port = 9999
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()
            logger.info(f"Server in ascolto su {self.host}:{self.port}")

    # ====================================================================
    # SERVER SOCKET
    # ====================================================================
    def _run_server(self) -> None:
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
                        continue

                    msg = json.loads(data)
                    logger.info(f"Messaggio ricevuto: {msg}")
                    response = {"status": "ok"}

                    # Comando locomozione gambe
                    if "cmd" in msg:
                        old_cmd = self.cmd.copy()
                        self.cmd = np.array(msg["cmd"], dtype=np.float32)
                        response["new_cmd"] = self.cmd.tolist()
                        logger.info(f"Nuovo comando locomozione: {old_cmd} -> {self.cmd}")

                    if "stop" in msg:
                        self.cmd = np.zeros(3, dtype=np.float32)
                        logger.info("STOP - Locomozione azzerata")

                    # Comando braccia/mani/vita
                    if "arm_cmd" in msg:
                        arm = np.array(msg["arm_cmd"], dtype=np.float32)
                        if len(arm) != len(self.other_indices):
                            response["status"] = "error"
                            response["msg"] = f"arm_cmd deve avere {len(self.other_indices)} valori"
                            logger.error(f"arm_cmd shape errata: {len(arm)} != {len(self.other_indices)}")
                        else:
                            self.arm_target = arm
                            response["arm_cmd"] = arm.tolist()
                            logger.info(f"Nuovo comando braccia: {arm}")

                    if "arm_reset" in msg:
                        self.arm_target = self.arm_default.copy()
                        logger.info("Braccia resettate alla pose default")

                    # Utility
                    if "reset" in msg:
                        mujoco.mj_resetData(self.model, self.data)
                        self.counter = 0
                        self.arm_target = self.arm_default.copy()
                        logger.info("RESET completo")

                    if "get_state" in msg:
                        response["pos"] = self.data.qpos[:3].tolist()
                        response["cmd"] = self.cmd.tolist()
                        logger.info(f"Stato: pos={response['pos']}")

                    conn.sendall(json.dumps(response).encode())
                    logger.info(f"Risposta inviata: {response}")

    # ====================================================================
    # POLICY RL — GAMBE
    # ====================================================================
    def _update_policy(self) -> None:
        qj   = self.data.qpos[7:7+self.ctrl_cfg.num_actions]
        dqj  = self.data.qvel[6:6+self.ctrl_cfg.num_actions]
        quat = self.data.qpos[3:7]
        omega = self.data.qvel[3:6]

        obs = build_observation(
            omega=omega,
            gravity=get_gravity_orientation(quat),
            cmd=self.cmd,
            qj=qj,
            dqj=dqj,
            action=self.action,
            phase=compute_gait_phase(self.counter, self.sim_cfg.simulation_dt),
            ang_vel_scale=self.obs_cfg.ang_vel_scale,
            dof_pos_scale=self.obs_cfg.dof_pos_scale,
            dof_vel_scale=self.obs_cfg.dof_vel_scale,
            cmd_scale=self.cmd_scale,
            default_angles=self.leg_default_angles,
            num_actions=self.ctrl_cfg.num_actions,
        )

        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze().astype(np.float32)
        self.leg_target = self.action * self.obs_cfg.action_scale + self.leg_default_angles
        self.target_dof_pos_full[self.leg_indices] = self.leg_target


    # ====================================================================
    # STEP — loop fisico 500Hz
    # ====================================================================
    def step(self) -> None:
        current_q  = self.data.qpos[7:7+self.total_dofs]
        current_dq = self.data.qvel[6:6+self.total_dofs]

        # ================================================================
        # GUADAGNI PD (FIXED)
        # ================================================================
        kp_full = np.zeros(self.total_dofs, dtype=np.float32)
        kd_full = np.zeros(self.total_dofs, dtype=np.float32)

        # GAMBE (policy RL)
        kp_full[self.leg_indices] = self.leg_kps
        kd_full[self.leg_indices] = self.leg_kds

        # OTHER (FIX: assegnazione, NON somma)
        kp_full[self.other_indices] = self.other_kp[self.other_indices]
        kd_full[self.other_indices] = self.other_kd[self.other_indices]

        # ================================================================
        # TARGET
        # ================================================================
        self.target_dof_pos_full[self.other_indices] = self.arm_target

        # ================================================================
        # PD CONTROL
        # ================================================================
        tau = pd_control(
            target_q=self.target_dof_pos_full,
            q=current_q,
            kp=kp_full,
            target_dq=np.zeros(self.total_dofs),
            dq=current_dq,
            kd=kd_full,
        )

        # ================================================================
        # 🚨 TORQUE CLIPPING (CRITICO)
        # ================================================================
        tau = np.clip(tau, -1.4, 1.4)

        self.data.ctrl[:self.total_dofs] = tau

        # ================================================================
        # STEP FISICO
        # ================================================================
        mujoco.mj_step(self.model, self.data)

        self.counter += 1

        # ================================================================
        # POLICY RL @ 50Hz
        # ================================================================
        if self.counter % self.sim_cfg.control_decimation == 0:
            if np.linalg.norm(self.cmd) > 0.05:
                self._update_policy()
            else:
                # FREEZE MODE
                self.data.qvel[:] = 0
                self.data.qacc[:] = 0
                self.data.qfrc_applied[:] = 0
                self.data.qpos[3:7] = [1, 0, 0, 0]
                self.data.qpos[7:7+self.ctrl_cfg.num_actions] = self.leg_default_angles

                self.leg_target = self.leg_default_angles.copy()
                self.target_dof_pos_full[self.leg_indices] = self.leg_default_angles

                mujoco.mj_forward(self.model, self.data)

    # ====================================================================
    # RUN — loop principale
    # ====================================================================
    def run(self) -> None:
        logger.info("Avvio loop principale di simulazione...")

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance = 5.0
            viewer.cam.elevation = -15
            viewer.cam.azimuth = 90
            logger.info("Visualizzatore avviato")

            step_count = 0
            while viewer.is_running():
                step_start = time.time()
                self.step()
                viewer.sync()
                step_count += 1

                # Log stato REALE dopo 100 step (fisica stabilizzata)
                if step_count == 100 or step_count == 500 or step_count == 1000:
                    self._log_runtime_state(step_count)

                elapsed = time.time() - step_start
                if self.model.opt.timestep > elapsed:
                    time.sleep(self.model.opt.timestep - elapsed)

            logger.info("Simulazione terminata")

    def _log_initial_state(self) -> None:
        """Debug: logga stato iniziale di tutti i DOF."""
        qpos = self.data.qpos[7:7+self.total_dofs]
        
        logger.info("=== STATO INIZIALE DOF ===")
        for i, idx in enumerate(self.other_indices):
            name = self.actuator_names[idx]
            qpos_val = qpos[idx]
            target_val = self.arm_target[i]
            diff = abs(qpos_val - target_val)
            status = "⚠️  MISMATCH" if diff > 0.01 else "✅ ok"
            logger.info(f"  [{idx:2d}] {name:<35} qpos={qpos_val:+.4f}  target={target_val:+.4f}  diff={diff:.4f}  {status}")
        logger.info("==========================")


    def _log_runtime_state(self, step_count) -> None:
        """Log stato reale dopo che la fisica si è stabilizzata."""
        qpos = self.data.qpos[7:7+self.total_dofs]
        logger.info(f"=== STATO REALE @ step {step_count} ===")
        for i, idx in enumerate(self.other_indices):
            name = self.actuator_names[idx]
            qpos_val = float(qpos[idx])
            target_val = float(self.arm_target[i])
            diff = abs(qpos_val - target_val)
            status = "⚠️  MISMATCH" if diff > 0.05 else "✅ ok"
            logger.info(f"  [{idx:2d}] {name:<35} qpos={qpos_val:+.4f}  target={target_val:+.4f}  diff={diff:.4f}  {status}")
        logger.info("==============================")