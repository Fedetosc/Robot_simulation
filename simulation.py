"""
G1 Robot Simulation with Mujoco and RL Policy.
"""

import time
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
    """
    Main simulation class for G1 robot walking.

    Handles:
        - Mujoco model loading and stepping
        - RL policy inference
        - PD control loop
        - Observation building
    """

    def __init__(
        self,
        sim_cfg: SimulationConfig,
        ctrl_cfg: ControlConfig,
        obs_cfg: ObservationConfig,
    ):
        """
        Initialize simulation.

        Args:
            sim_cfg: Simulation timing configuration
            ctrl_cfg: Controller gains and default positions
            obs_cfg: Observation scaling configuration
        """
        self.sim_cfg = sim_cfg
        self.ctrl_cfg = ctrl_cfg
        self.obs_cfg = obs_cfg

        # Convert configs to numpy arrays
        self.kps = np.array(ctrl_cfg.kps, dtype=np.float32)
        self.kds = np.array(ctrl_cfg.kds, dtype=np.float32)
        self.default_angles = np.array(ctrl_cfg.default_angles, dtype=np.float32)
        self.cmd_scale = np.array(obs_cfg.cmd_scale, dtype=np.float32)
        self.cmd = np.array(obs_cfg.cmd_init, dtype=np.float32)

        # State variables
        self.action = np.zeros(ctrl_cfg.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.obs = np.zeros(obs_cfg.num_obs, dtype=np.float32)
        self.counter = 0

        # Load Mujoco model
        self.model = mujoco.MjModel.from_xml_path(sim_cfg.xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = sim_cfg.simulation_dt

        # Load RL policy
        policy_path = Path(sim_cfg.policy_path)
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()

    def _compute_phase(self) -> float:
        """Compute current gait phase."""
        return compute_gait_phase(self.counter, self.sim_cfg.simulation_dt)

    def _build_observation(
        self,
        qj: NDArray[np.floating],
        dqj: NDArray[np.floating],
        quat: NDArray[np.floating],
        omega: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Build observation vector from robot state."""
        gravity = get_gravity_orientation(quat)
        phase = self._compute_phase()

        return build_observation(
            omega=omega,
            gravity=gravity,
            cmd=self.cmd,
            qj=qj,
            dqj=dqj,
            action=self.action,
            phase=phase,
            ang_vel_scale=self.obs_cfg.ang_vel_scale,
            dof_pos_scale=self.obs_cfg.dof_pos_scale,
            dof_vel_scale=self.obs_cfg.dof_vel_scale,
            cmd_scale=self.cmd_scale,
            default_angles=self.default_angles,
            num_actions=self.ctrl_cfg.num_actions,
        )

    def _infer_policy(self, obs: NDArray[np.floating]) -> NDArray[np.floating]:
        """Run RL policy inference."""
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        action = self.policy(obs_tensor).detach().numpy().squeeze()
        return action.astype(np.float32)

    def _apply_pd_control(self) -> None:
        """Apply PD control torque to actuators."""
        tau = pd_control(
            target_q=self.target_dof_pos,
            q=self.data.qpos[7:7+self.ctrl_cfg.num_actions],
            kp=self.kps,
            target_dq=np.zeros(self.ctrl_cfg.num_actions),
            dq=self.data.qvel[6:6+self.ctrl_cfg.num_actions],
            kd=self.kds,
        )
        self.data.ctrl[:self.ctrl_cfg.num_actions] = tau

    def _update_policy(self) -> None:
        """Update policy action and target joint positions."""
        # Read robot state
        qj = self.data.qpos[7:7+self.ctrl_cfg.num_actions]
        dqj = self.data.qvel[6:6+self.ctrl_cfg.num_actions]
        quat = self.data.qpos[3:7]
        omega = self.data.qvel[3:6]

        # Build observation and infer policy
        self.obs = self._build_observation(qj, dqj, quat, omega)
        self.action = self._infer_policy(self.obs)

        # Convert action to target joint positions
        self.target_dof_pos = self.action * self.obs_cfg.action_scale + self.default_angles

    def step(self) -> None:
        """Execute one simulation step."""
        # Apply PD control (runs at 500Hz)
        self._apply_pd_control()

        # Advance physics
        mujoco.mj_step(self.model, self.data)
        self.counter += 1

        # Update policy at control frequency (50Hz)
        if self.counter % self.sim_cfg.control_decimation == 0:
            self._update_policy()

    def run(self) -> None:
        """
        Run the simulation with visualizer.
        """
        print(f"Robot caricato: {self.model.nu} attuatori")
        print(f"Comando velocità: vx={self.cmd[0]}, vy={self.cmd[1]}, vyaw={self.cmd[2]}")
        print(
            f"Simulazione: {self.sim_cfg.simulation_duration}s a "
            f"{1/self.sim_cfg.simulation_dt:.0f}Hz fisico, "
            f"policy a {1/(self.sim_cfg.simulation_dt * self.sim_cfg.control_decimation):.0f}Hz"
        )

        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                # Camera setup
                viewer.cam.distance = 3.0
                viewer.cam.elevation = -15
                viewer.cam.azimuth = 90

                start = time.time()

                while viewer.is_running() and time.time() - start < self.sim_cfg.simulation_duration:
                    step_start = time.time()

                    self.step()

                    viewer.sync()

                    # Real-time pacing
                    elapsed = time.time() - step_start
                    sleep_time = self.model.opt.timestep - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("Interrotto dall'utente.")
        except Exception as e:
            print(f"Errore: {e}")
            import traceback
            traceback.print_exc()

        print("Simulazione completata.")


def run_simulation():
    """Entry point for running the simulation."""
    sim_cfg, ctrl_cfg, obs_cfg = load_config()
    sim = Simulation(sim_cfg, ctrl_cfg, obs_cfg)
    sim.run()
