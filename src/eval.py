import os
import shutil
import sys

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import PPO

from env import LiftCorrectionEnv


def make_video(frames_dir, output_path):
    """
    Create a video from images in frames_dir.
    """
    images = []
    # Sort files by frame id
    files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
    if not files:
        print(f"No frames found in {frames_dir}")
        return

    for filename in files:
        images.append(imageio.imread(os.path.join(frames_dir, filename)))
    if not images:
        print("No images to save.")
        return
    imageio.mimsave(output_path, images, fps=20)
    print(f"Video saved to {output_path}")


def evaluate():
    models_dir = "models"
    # Try the checkpoint model if final_model seems bad/missing
    # Prefer rl_model_2000_steps.zip if it exists
    candidates = ["rl_model_2000_steps", "final_model"]

    model_path = None
    for c in candidates:
        p = os.path.join(models_dir, c)
        if os.path.exists(p + ".zip"):
            model_path = p
            break

    if model_path is None:
        print("Model not found. Please train first.")
        return

    print(f"Loading model from: {model_path}")

    # Load model
    env = LiftCorrectionEnv()
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Try loading without env to see if it works
        model = PPO.load(model_path)

    num_episodes = 10
    success_count = 0

    eval_dir = "eval_results"
    if os.path.exists(eval_dir):
        shutil.rmtree(eval_dir)
    os.makedirs(eval_dir, exist_ok=True)

    print(f"Starting evaluation over {num_episodes} episodes...")

    best_episode_idx = -1

    for i in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False

        # Predict action
        action, _states = model.predict(obs, deterministic=True)

        print(f"\n--- Episode {i+1} ---")
        print(f"Noisy Pos: {obs['noisy_pos']}")
        print(f"Action (Correction): {action}")

        # Step
        obs, reward, terminated, truncated, info = env.step(action)

        is_success = info.get("is_success", False)
        print(f"Result: Success={is_success}, Reward={reward}")

        if is_success:
            success_count += 1
            if best_episode_idx == -1:
                best_episode_idx = i

        # Move frames to eval dir
        src_frames = "tmp_frames"
        dst_frames = os.path.join(eval_dir, f"episode_{i}")
        if os.path.exists(src_frames):
            shutil.copytree(src_frames, dst_frames)

    success_rate = success_count / num_episodes
    print(f"\nSuccess Rate: {success_rate * 100}% ({success_count}/{num_episodes})")

    # Generate video for the best (first successful) episode
    if best_episode_idx != -1:
        print(f"Generating video for Episode {best_episode_idx+1}...")
        frames_dir = os.path.join(eval_dir, f"episode_{best_episode_idx}")
        make_video(frames_dir, "best_controller_demo.mp4")
    else:
        print("No successful episodes to generate video.")
        # Generate video for the last episode anyway to debug
        frames_dir = os.path.join(eval_dir, f"episode_{num_episodes-1}")
        make_video(frames_dir, "failed_controller_demo.mp4")


if __name__ == "__main__":
    evaluate()
