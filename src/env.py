from utils import make_noisy_lift_env, move_ee_to, step_with_action, is_lift_success
import utils
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import os

# Patch save_frame to do nothing to speed up training and avoid disk usage
def no_op_save_frame(*args, **kwargs):
    # args[2] is frame_id, increment it
    if len(args) > 2:
        return args[2] + 1
    return 0

utils.save_frame = no_op_save_frame


class LiftCorrectionEnv(gym.Env):
    """
    Gymnasium environment for the robotic arm lift task with noisy observations.
    The agent observes the noisy position and camera image, and outputs a correction vector.
    The environment then runs the hardcoded controller with the corrected position.
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None):
        self.env = make_noisy_lift_env(
            add_noise=True,
            camera_names=("frontview",),
            image_size=(128, 128),
            flip_obs_images=True
        )
        self.action_space = spaces.Box(low=-0.15, high=0.15, shape=(3,), dtype=np.float32)
        
        # Observation space: Dictionary with image and noisy_pos
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            "noisy_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        })
        
        self.render_mode = render_mode
        self.current_obs = None
        self.true_cube_pos = None # For debugging/logging if needed (but not used by agent)
        self.start_pos_for_success = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_obs = self.env.reset()
        
        # Capture start pos for success check (Ground Truth)
        # We need the true pos to calculate reward, but we don't expose it to the agent.
        cube_bid = self.env.sim.model.body_name2id("cube_main")
        self.start_pos_for_success = self.env.sim.data.body_xpos[cube_bid].copy().astype(np.float32)
        
        image = self.current_obs["frontview_image"]
        noisy_pos = self.current_obs["cube_pos_noisy"].astype(np.float32)
        
        return {
            "image": image,
            "noisy_pos": noisy_pos
        }, {}
        
    def step(self, action):
        """
        Action is a 3D correction vector.
        We apply this correction to the noisy position, then run the scripted controller.
        """
        noisy_pos = self.current_obs["cube_pos_noisy"]
        corrected_pos = noisy_pos + action
        
        # --- SCRIPTED CONTROLLER EXECUTION ---
        cam_key = "frontview_image"
        # Helper to ignore frame saving overhead
        frames_dir = "tmp_frames"
        import shutil
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir, exist_ok=True)
        frame_id = 0
        action_dim = self.env.action_dim
        obs = self.current_obs
        
        target_above = corrected_pos.copy(); target_above[2] += 0.15
        target_grasp = corrected_pos.copy(); # target_grasp[2] += 0.02
        target_lift  = corrected_pos.copy(); target_lift[2]  += 0.25
        
        # 1. Open
        action_open = np.zeros(action_dim); action_open[-1] = -1.0
        obs, frame_id = step_with_action(self.env, action_open, 20, obs, cam_key, frame_id, frames_dir)
        
        # 2. Above
        obs, frame_id = move_ee_to(self.env, obs, target_above, -1.0, 100, action_dim, cam_key, frame_id, frames_dir, kp=8.0, ki=0.0, kd=1.0, max_delta=0.1)
        
        # 3. Grasp
        obs, frame_id = move_ee_to(self.env, obs, target_grasp, -1.0, 250, action_dim, cam_key, frame_id, frames_dir, kp=[10.0, 10.0, 12.0], ki=0.0, kd=0.2, max_delta=0.1)
        
        # 4. Close
        action_close = np.zeros(action_dim); action_close[-1] = 1.0
        obs, frame_id = step_with_action(self.env, action_close, 40, obs, cam_key, frame_id, frames_dir)
        
        # 5. Lift
        obs, frame_id = move_ee_to(self.env, obs, target_lift, 1.0, 200, action_dim, cam_key, frame_id, frames_dir, kp=10.0, ki=0.0, kd=1.0, max_delta=0.1)
        
        self.current_obs = obs
        
        # --- SUCCESS CHECK ---
        cube_bid = self.env.sim.model.body_name2id("cube_main")
        current_true_pos = self.env.sim.data.body_xpos[cube_bid].copy()
        
        dz = current_true_pos[2] - self.start_pos_for_success[2]
        xy_shift = np.linalg.norm(current_true_pos[:2] - self.start_pos_for_success[:2])
        
        success = (dz >= 0.10) and (xy_shift <= 0.1)
        
        # --- REWARD SHAPING ---
        # Calculate distance between Corrected Target and True Object Position
        dist_error = np.linalg.norm(corrected_pos - self.start_pos_for_success)
        
        reward = 0.0
        if success:
            reward += 2.0
        
        # Distance penalty (scaled so 10cm error = -1.0 reward)
        reward -= (dist_error * 10.0)
        
        # Optional: Clip reward to avoid extreme negatives
        reward = max(reward, -2.0)
        
        terminated = True
        truncated = False
        
        return {
            "image": obs["frontview_image"],
            "noisy_pos": obs["cube_pos_noisy"].astype(np.float32)
        }, reward, terminated, truncated, {"is_success": success}

    def close(self):
        self.env.close()
