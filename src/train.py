import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import utils
from env import LiftCorrectionEnv

def no_op_save_frame(*args, **kwargs):
    if len(args) > 2:
        return args[2] + 1
    return 0

utils.save_frame = no_op_save_frame

def train():
    log_dir = "logs"
    models_dir = "models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    env = LiftCorrectionEnv()
    env = Monitor(env, log_dir)
    
    eval_env = LiftCorrectionEnv()
    eval_env = Monitor(eval_env, log_dir)

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        tensorboard_log=log_dir,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[128, 128],
                vf=[128, 128]
            ),
            activation_fn=torch.nn.ReLU,
        )
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=models_dir,
        name_prefix="rl_model"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=500,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    print("Starting training...")
    total_timesteps = 10000 
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])

    model.save(os.path.join(models_dir, "final_model"))
    print("Training finished. Model saved.")

if __name__ == "__main__":
    train()
