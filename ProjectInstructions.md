
# Task
Your task is to develop a controller for a robotic arm to lift an object. Such control is not difficult if the
accurate location of the object is known. However, in real world scenarios, we often have noise in
measurement and thus the controller won’t have precise location information. Using reinforcement
learning, you will develop a controller that has access to both noise perturbed location information and
camera images. The controller can use the images to adjust the location information to complete the lift
task.
## Robotic Simulation
The robotic simulation class and functions are in the “utils.py” file.
The notebook (“Robosuite Demo.ipynb”) shows how to use the simulation environment and gives a
waypoint-based controller. The controller can complete the task if noise is turned off. But it often fails
when noise is present.
## Objective
Design a reinforcement learning process to train a neural network that can be used to adjust the control
code in the example so that the arm can lift the object with noisy location information.
Your design must meet the following requirements:
1. The neural network adjustment must be trained using reinforcement learning, not supervised
learning.
2. The true location information is available in the simulation code. But you cannot use the true
location information in any part of training or testing process. This further prevents
approaches such as training a neural network to predict the accurate location by supervised
learning because you cannot use the true location information as your training target.
3. Each time you create a new simulation environment, the location of the object and the amount of
noise perturbation are different. Your neural network and adjustment should work for any object
location and noise perturbation.
You are free to use any neural network architecture and any reinforcement learning algorithm.
To test your neural network and control algorithm, create 10 simulation environments (each with random
object location and noise). Run your control algorithm trained by reinforcement learning in each of the
environment. Calculate the success rate (i.e., # of environment among the 10 in which your control can
successfully complete the task.
Record videos for one environment in which your controller gives the best result.
## Submission
Code and model: organize your code in a directory named “src”. It should include:
1) Your controller code together with the neural network for making the adjustment. The code
should allow downloading your trained model from remote servers(like google drive) or local path so that we can run the controller in a new environment without training.
2) Code implementing the reinforcement training scheme you designed.
3) A README file providing:
a. A short description of the codebase
b. Instruction for running your code to conduct reinforcement learning and instruction for
using your trained controller

Vidoe:
1) A demo video showing the best attempt by your controller. The Robosuite Demo.ipynb has a video generation cell for your help.
Writeup: write a 4-page report on what you did, including
1) the reinforcement learning scheme you designed and a justification on why you think it may
work;
2) the success rate of the trained model;
3) the information on the true object location and the noisy object location for the environments in
your example video;
4) Analysis of the reason behind the result and any lesson learned.
(Your method may not train a successful policy/controller, but it needs to make sense in a theoretical way.
Innovative solution, theoretically soundness and proper validation are the main considerations for
grading.)
Save the writing in a pdf file. (use latex for it)