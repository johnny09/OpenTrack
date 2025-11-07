import os
import json

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

import logging as python_logging

LOGGER = python_logging.getLogger()
LOGGER.setLevel(python_logging.INFO)

from absl import logging

logging.set_verbosity(logging.INFO)

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from tqdm import tqdm
import torch

from src.envs.g1.g1_tracking_env import default_config as g1_default_config
from src.envs.g1.play_g1_tracking_env import PlayG1TrackingEnv

from src.envs.adamsp.adamsp_tracking_env import default_config as adamsp_default_config
from src.envs.adamsp.play_adamsp_tracking_env import PlayAdamSPTrackingEnv


@dataclass
class Args:
    exp_name: str
    robot: str = "PndAdamSP"  # choose from PndAdamSP, UnitreeG1
    play_ref_motion: bool = False
    use_viewer: bool = False  # passive viewer (with display)
    use_renderer: bool = False  # renderer with video (headless mode)


@dataclass
class State:
    info: dict
    obs: dict


def get_latest_ckpt(path: Path) -> Path | None:
    ckpts = [ckpt for ckpt in path.glob("*") if not ckpt.name.endswith(".json")]
    ckpts.sort(key=lambda x: int(x.name))
    return ckpts[-1] if ckpts else None


def play(args: Args):
    robot = args.robot

    if robot == "PndAdamSP":
        task_cfg = adamsp_default_config()
    elif robot == "UnitreeG1":
        task_cfg = g1_default_config()
    else:
        raise ValueError(f"Unknown robot: {robot}")

    env_cfg = task_cfg.env_config

    config_path = (
        Path(__file__).parent
        / "experiments"
        / args.exp_name
        / "checkpoints"
        / "config.json"
    )
    with open(config_path, "r") as f:
        config = json.load(f)
    del config["env_config"]["reference_traj_config"]
    env_cfg.update(config["env_config"])

    # env_cfg.reference_traj_config.name = {"lafan1": ["dance1_subject2"]}

    if robot == "PndAdamSP":
        env = PlayAdamSPTrackingEnv(
            terrain_type=env_cfg.terrain_type,
            config=env_cfg,
            play_ref_motion=args.play_ref_motion,
            use_viewer=args.use_viewer,
            use_renderer=args.use_renderer,
            exp_name=args.exp_name,
        )
    elif robot == "UnitreeG1":
        env = PlayG1TrackingEnv(
            terrain_type=env_cfg.terrain_type,
            config=env_cfg,
            play_ref_motion=args.play_ref_motion,
            use_viewer=args.use_viewer,
            use_renderer=args.use_renderer,
            exp_name=args.exp_name,
        )

    ckpt_path = Path(__file__).parent / "experiments" / args.exp_name / "checkpoints"
    latest_ckpt = get_latest_ckpt(ckpt_path)
    if latest_ckpt is None:
        raise FileNotFoundError("No checkpoint found.")

    policy_path = latest_ckpt / "policy.pt"
    policy_jit = torch.jit.load(policy_path, map_location="cpu")
    state = env.reset()

    len_traj = (
        env.th.traj.data.qpos.shape[0] - len(env_cfg.reference_traj_config.name) - 1
    )
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    import time

    time.sleep(3)

    saved_joint_torque = []
    saved_joint_velocity = []
    for i in tqdm(range(len_traj)):
        with torch.no_grad():
            action = (
                policy_jit(
                    torch.from_numpy(
                        state.obs["state"].reshape(1, -1).astype(np.float32)
                    )
                )
                .cpu()
                .numpy()
            )
        state = env.step(state, action)
        if i < 0.15 * len_traj:
            # 存储 joint_torque
            joint_torque = np.array(state.info["joint_torque"])
            if joint_torque.ndim == 1:  # 一组
                saved_joint_torque.append(joint_torque)
            else:
                for jt in joint_torque:
                    saved_joint_torque.append(jt)

            # 存储 joint_velocity
            joint_velocity = np.array(state.info["joint_velocity"])
            if "saved_joint_velocity" not in locals():
                saved_joint_velocity = []
            if joint_velocity.ndim == 1:
                saved_joint_velocity.append(joint_velocity)
            else:
                for jv in joint_velocity:
                    saved_joint_velocity.append(jv)
        else:
            import matplotlib.pyplot as plt

            saved_joint_torque = np.array(saved_joint_torque)
            saved_joint_velocity = np.array(saved_joint_velocity)
            # Scatter plot of (joint_velocity, joint_torque) for each joint, subplot layout as before
            layout = [6, 6, 3, 7, 7]
            num_row = len(layout)
            num_col = max(layout)
            fig, axs = plt.subplots(
                num_row,
                num_col,
                figsize=(3 * num_col, 2.5 * num_row),
                sharex=False,
                sharey=False,
            )

            joint_idx = 0
            for row, row_ncol in enumerate(layout):
                for col in range(row_ncol):
                    if joint_idx >= saved_joint_torque.shape[1]:
                        axs[row, col].axis("off")
                        continue
                    axs[row, col].scatter(
                        saved_joint_velocity[:, joint_idx],
                        saved_joint_torque[:, joint_idx],
                        s=2,
                        alpha=0.7,
                    )
                    axs[row, col].set_xlabel(f"Joint {joint_idx} velocity")
                    axs[row, col].set_ylabel(f"Torque")
                    axs[row, col].grid(True)
                    joint_idx += 1
                # Turn off any unused subplots in this row
                for col in range(row_ncol, num_col):
                    axs[row, col].axis("off")

            plt.tight_layout()
            plt.show()
            env.close()
            exit()


if __name__ == "__main__":
    args = tyro.cli(Args)
    play(args)
