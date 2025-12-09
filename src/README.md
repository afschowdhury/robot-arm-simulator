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

The implementation is fully compliant with project requirements:
- Uses reinforcement learning (PPO) with sparse rewards
- Does NOT use true location information in reward calculation
- Only uses task success/failure as learning signal
- Learns from camera images and noisy position observations

Directory Structure:
- src/env.py: The Gym wrapper around the Robosuite environment
- src/train.py: Script to train the PPO agent
- src/eval.py: Script to evaluate the trained agent and generate a demo video
- src/model_utils.py: Utilities for loading models
- utils.py: Helper functions for the simulation environment

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
- torch

3. Usage
--------

Training:
    python src/train.py
    
    This will train the model for 10,000 episodes and save it to `models/final_model.zip`.
    Logs are saved to `logs/`.
    Training uses sparse rewards based only on task success/failure.
    Evaluation is performed every 500 episodes during training.

Evaluation:
    python src/eval.py
    
    This will load `models/final_model` and run 10 evaluation episodes.
    It prints the success rate and saves a video of the best episode to `best_controller_demo.mp4`.

4. Trained Models
-----------------
URL: [INSERT_LINK_HERE]

To use a downloaded model:
1. Download the zip file
2. Place it in `models/final_model.zip`
3. Run `python src/eval.py`

5. Implementation Details
-------------------------
- Policy Network: Two-layer MLP (128, 128) processing image features and noisy position
- Value Network: Two-layer MLP (128, 128)
- CNN: NatureCNN architecture (default in stable-baselines3)
- Action Space: Box(-0.15, 0.15, shape=(3,)) for 3D position correction
- Observation Space: Dict with 128x128x3 RGB image and 3D noisy position vector
- Reward: Sparse (2.0 for success, 0.0 for failure)
- Training Episodes: 10,000
- PPO Hyperparameters:
  - learning_rate: 3e-4
  - n_steps: 2048
  - batch_size: 64
  - n_epochs: 10
  - gamma: 0.99
  - gae_lambda: 0.95
  - ent_coef: 0.01
  - clip_range: 0.2
