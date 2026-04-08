import time
import socket
import threading
import json
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
from numpy.typing import NDArray

from config import load_config, ControlConfig, ObservationConfig, SimulationConfig
from controller import pd_control
from observer import build_observation, compute_gait_phase, get_gravity_orientation


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

        # Conversione parametri
        self.kps = np.array(ctrl_cfg.kps, dtype=np.float32)
        self.kds = np.array(ctrl_cfg.kds, dtype=np.float32)
        self.default_angles = np.array(ctrl_cfg.default_angles, dtype=np.float32)
        self.cmd_scale = np.array(obs_cfg.cmd_scale, dtype=np.float32)
        self.cmd = np.array(obs_cfg.cmd_init, dtype=np.float32)

        # Variabili di stato
        self.action = np.zeros(ctrl_cfg.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.counter = 0

        # Caricamento Mujoco
        self.model = mujoco.MjModel.from_xml_path(sim_cfg.xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = sim_cfg.simulation_dt

        # Caricamento RL Policy
        self.policy = torch.jit.load(Path(sim_cfg.policy_path))
        self.policy.eval()

        # Avvio Server Socket se richiesto
        if start_server:
            self.host = "localhost"
            self.port = 9999
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()
            print(f"[*] Server in ascolto su {self.host}:{self.port}")

    def _run_server(self):
        """Thread per ricevere comandi JSON dal commander.py"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen()
            while True:
                conn, addr = s.accept()
                with conn:
                    data = conn.recv(1024).decode()
                    if not data: continue
                    msg = json.loads(data)
                    response = {"status": "ok"}

                    if "cmd" in msg:
                        self.cmd = np.array(msg["cmd"], dtype=np.float32)
                        response["new_cmd"] = self.cmd.tolist()
                    if "stop" in msg:
                        self.cmd = np.zeros(3, dtype=np.float32)
                    if "reset" in msg:
                        mujoco.mj_resetData(self.model, self.data)
                        self.counter = 0
                    if "get_state" in msg:
                        response["pos"] = self.data.qpos[:3].tolist()
                    
                    conn.sendall(json.dumps(response).encode())

    def _update_policy(self) -> None:
        """Inference della rete neurale"""
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

    def step(self) -> None:
        """Passo di simulazione: 500Hz PD, 50Hz Policy"""
        # PD Control (Torque)
        self.data.ctrl[:self.ctrl_cfg.num_actions] = pd_control(
            target_q=self.target_dof_pos,
            q=self.data.qpos[7:7+self.ctrl_cfg.num_actions],
            kp=self.kps,
            target_dq=np.zeros(self.ctrl_cfg.num_actions),
            dq=self.data.qvel[6:6+self.ctrl_cfg.num_actions],
            kd=self.kds,
        )

        mujoco.mj_step(self.model, self.data)
        self.counter += 1

        # Control Decimation (Policy Update)
        if self.counter % self.sim_cfg.control_decimation == 0:
            self._update_policy()

    def run(self) -> None:
        """Loop principale con visualizzatore"""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Configurazione camera
            viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 5.0, -15, 90
            
            while viewer.is_running():
                step_start = time.time()
                self.step()
                viewer.sync()

                # Sincronizzazione tempo reale
                elapsed = time.time() - step_start
                if self.model.opt.timestep > elapsed:
                    time.sleep(self.model.opt.timestep - elapsed)


def run_simulation(is_server: bool = False):
    sim_cfg, ctrl_cfg, obs_cfg = load_config()
    
    # Se server, ignora il cammino automatico iniziale
    if is_server:
        obs_cfg.cmd_init = [0.0, 0.0, 0.0]
        
    sim = Simulation(sim_cfg, ctrl_cfg, obs_cfg, start_server=is_server)
    sim.run()