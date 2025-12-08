## Project Specification: Robotic Arm Control via Reinforcement Learning

1. Project Overview
   Goal: Develop a controller for a robotic arm to lift an object. While the simulation provides true location data, real-world sensors are noisy. You must train a neural network using Reinforcement Learning (RL) to adjust the control inputs based on camera images and noisy location data to successfully lift the object.

Core Challenge: The existing waypoint-based controller works perfectly without noise but fails when measurement noise is introduced. You must fix this using RL.

2. Provided Resources
   utils.py: Contains the robotic simulation class and functions.

Robosuite Demo.ipynb: Demonstrates how to use the environment and includes the baseline waypoint-based controller.

3. Strict Constraints & Rules
   Methodology: You MUST use Reinforcement Learning. You generally cannot use Supervised Learning.

Data Usage (CRITICAL):

You have access to noisy location data and camera images.

Although the simulation code contains the "true" location, you cannot use the true location information in any part of the training or testing process.

You cannot train a network to predict the true location (Supervised Learning) using the true location as a target.

Generalization: The system must work on random object locations and random noise perturbations. It must not be overfit to a single environment seed.

Architecture: You are free to use any Neural Network architecture and RL algorithm (e.g., PPO, DQN, SAC).

4. Implementation Steps
   Phase 1: Setup & Design

Analyze the Baseline: Run the provided Robosuite Demo.ipynb to understand the inputs and why the controller fails with noise.

Design the State Space: The input to your agent will be the noisy location and camera images.

Design the Action Space: The neural network should output an adjustment to the control code (e.g., an offset vector) rather than controlling the joint torques directly, though you have flexibility here.

Design the Reward Function: Create a reward signal based on whether the object was successfully lifted.

Phase 2: Training (The "src" Directory)

Create a directory named src containing:

RL Training Code: Implement the training loop.

Model Architecture: Define the neural network.

Model Loader: Implement functionality to download/load the trained model so it can run in a new environment without retraining.

Phase 3: Validation & Testing

Test Environment: Create 10 distinct simulation environments. Each must have a random object location and random noise.

Execution: Run the trained RL controller in all 10 environments.

Metrics: Calculate the success rate (Number of successes / 10).


5. Deliverables
   A. Code Structure

Organize the code into a directory named src containing:

Controller code + Neural Network.

Training implementation.

A README.txt inside src with:

Description of the codebase.

Instructions to run training and inference.

Links to download the saved trained models (Do not upload large model files directly; host them on Drive/Dropbox).

B. Report (PDF - should use LaTex for formatting)

Write a 4-page report including:

Methodology: The RL scheme designed and theoretical justification for why it works.

Results: The success rate (x/10).

Analysis:

Show the true object location vs. noisy location for the example video environment.

Analyze why the result occurred and lessons learned.

Note: Theoretical soundness is graded even if the policy isn't perfect.
