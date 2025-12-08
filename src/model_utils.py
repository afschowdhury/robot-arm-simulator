import os
from stable_baselines3 import PPO
import gymnasium as gym
# We need to import the Env class so that pickle knows about it if saved with the model, 
# though SB3 usually saves just weights. 
# But loading might require re-creating the env.

def load_trained_model(model_path, env=None):
    """
    Load a trained PPO model.
    
    Args:
        model_path (str): Path to the .zip file.
        env (gym.Env, optional): The environment to run the model in. 
                                 If None, it's loaded without an env (but prediction works).
    
    Returns:
        model: Loaded SB3 PPO model.
    """
    if not os.path.exists(model_path):
        if os.path.exists(model_path + ".zip"):
            model_path += ".zip"
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
            
    # Load model
    # We pass custom_objects if needed, but standard PPO shouldn't need it.
    model = PPO.load(model_path, env=env)
    return model

def download_model(url, save_path):
    """
    Placeholder for downloading model from a URL.
    In a real scenario, use requests or urllib.
    """
    import urllib.request
    print(f"Downloading model from {url}...")
    urllib.request.urlretrieve(url, save_path)
    print("Download complete.")

