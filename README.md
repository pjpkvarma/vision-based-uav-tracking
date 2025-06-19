# 🛰️ Active Aerial Object Tracking and Pursuit using Reinforcement Learning

This repository contains the official implementation of my **Master’s Thesis** titled:

> **"Active Aerial Object Tracking and Pursuit using Reinforcement Learning"**  
> *by Jagadeswara Pavan Kumar Varma Pothuri*

---

## Project Overview

We propose a **vision-based reinforcement learning** framework for enabling UAVs to track and pursue dynamic aerial targets. The system integrates:

- **YOLO** for initial object detection  
- **KCF (Kernelized Correlation Filter)** with APCE validation for efficient tracking  
- **AirSim + Gym-compatible environment** for realistic drone simulation  
- **PPO (Proximal Policy Optimization)** algorithm from Stable-Baselines3 for training pursuit behavior  
- **WandB integration** for tracking training metrics and model evaluation  

---

## Features

- ✅ Dual-drone simulation (Target + Chaser) in AirSim
- ✅ Hybrid visual tracking: YOLO for detection, KCF for frame-to-frame localization
- ✅ APCE-based tracker reinitialization
- ✅ Constant Velocity Estimator (CVE) for future state prediction
- ✅ Sine-path generator for target motion dynamics
- ✅ PPO training with full logging and evaluation support

---

## Repository Structure

```bash
drone-vision-rl/
├── drone_env.py               # Custom Gym environment (AirSim + hybrid perception)
├── detect_drone.py            # YOLOv8 + KCF tracker integration
├── kcf.py                     # HOG + KCF tracker with APCE computation
├── path_generator.py          # Sine path generator + trajectory plotting
├── train.py                   # RL training script (PPO + WandB)
├── README.md                  # Project documentation

