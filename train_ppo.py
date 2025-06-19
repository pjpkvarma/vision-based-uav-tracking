import os
from pathlib import Path

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from drone_env import DroneEnv

import wandb
from wandb.integration.sb3 import WandbCallback

# Directory for logs and model checkpoints
log_dir = Path("./dqn_logs")
log_dir.mkdir(parents=True, exist_ok=True)

# WandB config
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": int(1e6),
    "env_name": "drone-track",
    "vf_coef": 0.5,
    "batch_size": 256,
    "n_steps": 1024,
    "learning_rate": 1e-4,
    "max_grad_norm": 0.5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.05
}

# Initialize environment and monitoring
env = DroneEnv()
env = Monitor(env, filename=str(log_dir / "monitor"))

# Initialize WandB run
run = wandb.init(
    project="track discrete old code",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)

# Model save/load paths
model_path = log_dir / "tb_logs"
pretrained_model_path = "/data/users/abhatt4/old_codes_descrete/dqn_logs/ppo_model_260000_steps.zip"
final_model_path = log_dir / "ppo_drone_final"

# PPO policy architecture
policy_kwargs = dict(net_arch=[128, 128])

# Load pretrained model if available
if pretrained_model_path and os.path.exists(pretrained_model_path):
    model = PPO.load(
        pretrained_model_path,
        env=env,
        tensorboard_log=str(model_path) + f"runs/{run.id}",
        device="cpu",
        reset_num_timesteps=False
    )
else:
    model = PPO(
        "MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(model_path) + f"runs/{run.id}",
        device="cpu"
    )

# Callbacks for saving models and evaluation
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=str(log_dir),
    name_prefix="ppo_model"
)

eval_callback = EvalCallback(
    env,
    best_model_save_path=str(log_dir / "best_model"),
    log_path=str(log_dir),
    eval_freq=5000,
    deterministic=True,
    render=False
)

# Train the model
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=[checkpoint_callback, eval_callback]
)

# Save final model
model.save(str(final_model_path))
