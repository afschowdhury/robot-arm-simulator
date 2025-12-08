Robotic Arm Lift Controller with Reinforcement Learning
=====================================================

1. Description
--------------
This codebase implements a Residual Reinforcement Learning agent to correct noisy object location measurements for a robotic arm lift task. 
The agent uses a PPO (Proximal Policy Optimization) policy that takes two inputs:
- Frontview camera image (128x128 RGB)
- Noisy object position (x, y, z)

It outputs a 3D correction vector (dx, dy, dz) which is added to the noisy position. 
This corrected position is then used by a hardcoded waypoint-based controller to execute the pick-and-place task.

Directory Structure:
- src/env.py: The Gym wrapper around the Robosuite environment.
- src/train.py: Script to train the PPO agent.
- src/eval.py: Script to evaluate the trained agent and generate a demo video.
- src/model_utils.py: Utilities for loading models.
- src/utils.py: (Symlink/Import) Helper functions provided by the course.

2. Installation
---------------
Use the `dl-assign` conda environment.
Dependencies:
- stable-baselines3
- gymnasium
- shimmy
- robosuite
- mujoco
- imageio[ffmpeg]

(These should be installed via the provided instructions).

3. Usage
--------

Training:
    python src/train.py
    
    This will train the model for 2000 episodes and save it to `models/final_model.zip`.
    Logs are saved to `logs/`.

Evaluation:
    python src/eval.py
    
    This will load `models/final_model` and run 10 evaluation episodes.
    It prints the success rate and saves a video of the best episode to `best_controller_demo.mp4`.

4. Trained Models
-----------------
(Add link to Google Drive / Dropbox here after training)
URL: [INSERT_LINK_HERE]

To use a downloaded model:
1. Download the zip file.
2. Place it in `models/final_model.zip`.
3. Run `python src/eval.py`.

