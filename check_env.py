import torch
import stable_baselines3
import numpy as np
import gymnasium
import shimmy
import os

print(f"Torch: {torch.__version__}")
print(f"SB3: {stable_baselines3.__version__}")
print(f"Numpy: {np.__version__}")
print(f"Gymnasium: {gymnasium.__version__}")
print(f"Shimmy: {shimmy.__version__}")

model_path = "models/final_model.zip"
if os.path.exists(model_path):
    print(f"Model found at {model_path}")
    
    try:
        import zipfile
        with zipfile.ZipFile(model_path, 'r') as archive:
            print("Files in archive:", archive.namelist())
    except Exception as e:
        print("Error reading zip:", e)
else:
    print("Model not found!")
