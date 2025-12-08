import sys
import os

# Add parent directory to path to import utils (utils.py is in project root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import utils
from env import LiftCorrectionEnv

# Patch save_frame to do nothing to speed up training and avoid disk usage
def no_op_save_frame(*args, **kwargs):
    # args[2] is frame_id, increment it
    if len(args) > 2:
        return args[2] + 1
    return 0

utils.save_frame = no_op_save_frame

def train():
    # Create logs directory
    log_dir = "logs"
    models_dir = "models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Create environment
    env = LiftCorrectionEnv()
    env = Monitor(env, log_dir)

    # Define PPO model
    # We use MultiInputPolicy because observation is a Dict (image + vector)
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=128,          # Short horizon per update, but each step is ONE EPISODE (simulated)
                              # Actually, one env step = one episode. 
                              # So n_steps=128 means 128 episodes per update.
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log=log_dir,
        policy_kwargs=dict(
            net_arch=[64, 64], # MLP part
            # CNN part defaults to NatureCNN
        )
    )

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=models_dir,
        name_prefix="rl_model"
    )

    print("Starting training...")
    # Train
    # Since each step is a full episode, 10,000 steps = 10,000 episodes.
    # This should be plenty for learning a 3D offset.
    total_timesteps = 2000 
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # Save final model
    model.save(os.path.join(models_dir, "final_model"))
    print("Training finished. Model saved.")

if __name__ == "__main__":
    train()

