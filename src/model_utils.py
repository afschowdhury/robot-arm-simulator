import os
from stable_baselines3 import PPO
import gymnasium as gym

def load_trained_model(model_path, env=None):
    if not os.path.exists(model_path):
        if os.path.exists(model_path + ".zip"):
            model_path += ".zip"
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
            
    model = PPO.load(model_path, env=env)
    return model

def download_model(url, save_path):
    import urllib.request
    print(f"Downloading model from {url}...")
    urllib.request.urlretrieve(url, save_path)
    print("Download complete.")
