import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import make_noisy_lift_env, move_ee_to, step_with_action, is_lift_success
import utils
import gymnasium as gym
import numpy as np
from gymnasium import spaces

def no_op_save_frame(*args, **kwargs):
    if len(args) > 2:
        return args[2] + 1
    return 0

utils.save_frame = no_op_save_frame


class LiftCorrectionEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None):
        self.env = make_noisy_lift_env(
            add_noise=True,
            camera_names=("frontview",),
            image_size=(128, 128),
            flip_obs_images=True
        )
        self.action_space = spaces.Box(low=-0.15, high=0.15, shape=(3,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            "noisy_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        })
        
        self.render_mode = render_mode
        self.current_obs = None
        self.true_cube_pos = None
        self.start_pos_for_success = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_obs = self.env.reset()
        
        cube_bid = self.env.sim.model.body_name2id("cube_main")
        self.start_pos_for_success = self.env.sim.data.body_xpos[cube_bid].copy().astype(np.float32)
        
        image = self.current_obs["frontview_image"]
        noisy_pos = self.current_obs["cube_pos_noisy"].astype(np.float32)
        
        return {
            "image": image,
            "noisy_pos": noisy_pos
        }, {}
        
    def step(self, action):
        noisy_pos = self.current_obs["cube_pos_noisy"]
        corrected_pos = noisy_pos + action
        
        cam_key = "frontview_image"
        frames_dir = "tmp_frames"
        import shutil
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir, exist_ok=True)
        frame_id = 0
        action_dim = self.env.action_dim
        obs = self.current_obs
        
        target_above = corrected_pos.copy(); target_above[2] += 0.15
        target_grasp = corrected_pos.copy()
        target_lift  = corrected_pos.copy(); target_lift[2]  += 0.25
        
        action_open = np.zeros(action_dim); action_open[-1] = -1.0
        obs, frame_id = step_with_action(self.env, action_open, 20, obs, cam_key, frame_id, frames_dir)
        
        obs, frame_id = move_ee_to(self.env, obs, target_above, -1.0, 100, action_dim, cam_key, frame_id, frames_dir, kp=8.0, ki=0.0, kd=1.0, max_delta=0.1)
        
        obs, frame_id = move_ee_to(self.env, obs, target_grasp, -1.0, 250, action_dim, cam_key, frame_id, frames_dir, kp=[10.0, 10.0, 12.0], ki=0.0, kd=0.2, max_delta=0.1)
        
        action_close = np.zeros(action_dim); action_close[-1] = 1.0
        obs, frame_id = step_with_action(self.env, action_close, 40, obs, cam_key, frame_id, frames_dir)
        
        obs, frame_id = move_ee_to(self.env, obs, target_lift, 1.0, 200, action_dim, cam_key, frame_id, frames_dir, kp=10.0, ki=0.0, kd=1.0, max_delta=0.1)
        
        self.current_obs = obs
        
        cube_bid = self.env.sim.model.body_name2id("cube_main")
        current_true_pos = self.env.sim.data.body_xpos[cube_bid].copy()
        
        dz = current_true_pos[2] - self.start_pos_for_success[2]
        xy_shift = np.linalg.norm(current_true_pos[:2] - self.start_pos_for_success[:2])
        
        success = (dz >= 0.10) and (xy_shift <= 0.1)
        
        reward = 2.0 if success else 0.0
        
        terminated = True
        truncated = False
        
        return {
            "image": obs["frontview_image"],
            "noisy_pos": obs["cube_pos_noisy"].astype(np.float32)
        }, reward, terminated, truncated, {"is_success": success}

    def close(self):
        self.env.close()
