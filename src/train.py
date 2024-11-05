from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from easydict import EasyDict
import torch.nn as nn
from environment.env import BoosterEnv
from training.booster_tracker import Sb3BoosterTracker
from common.utils import create_training_folder
from common.yaml_helper import load_config, show_dict, save_config
#from common.logger import *
from typing import Callable, Optional
from stable_baselines3.common.utils import get_linear_fn
import logging
import os

def _policy_generator(config: EasyDict) -> Optional[dict]:
    """
    Generates the policy network architecture based on config specifications.

    Args:
        config (EasyDict): Configuration dictionary for policy parameters, including
                           layer count, nodes per layer, and activation function.

    Returns:
        dict: A dictionary defining the policy network architecture for PPO or None for default.
    """
    if config.layers is None or config.nodes is None:
        # Return default network if no custom layers/nodes are defined
        return None

    # Define architecture: a list with `layers` number of `nodes`
    net_arch = [config.nodes for _ in range(config.layers)]

    # Map activation function name to its PyTorch equivalent
    activation_fn = config.get("activation", "tanh")
    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU
    }.get(activation_fn, nn.Tanh)  # Default to Tanh if invalid name

    return {"net_arch": net_arch, "activation_fn": activation_fn}


def _initialize_ppo(config: EasyDict, env: BoosterEnv) -> PPO:
    """
    Initializes a PPO model based on training configuration.

    Args:
        config (EasyDict): Configuration parameters for PPO.
        env (BoosterEnv): The environment for training the PPO agent.

    Returns:
        PPO: Initialized PPO model ready for training.
    """
    # Calculate linear learning rate schedule for PPO if needed
    end_fraction = config.get("lr_end_episodes", 1000) / config["episodes"]

    # Configure the PPO model
    model = PPO(
        config.policy,
        env,
        policy_kwargs=_policy_generator(config),
        verbose=config.verbose,
        learning_rate=get_linear_fn(config.lr_start, config.lr_end, 0.01),  #config.lr or 3e-4, #
        gamma=config.gamma or 0.99,
        normalize_advantage=config.normalize_advantage,
        device=config.device or "auto",
        batch_size=config.batch_size or 64,
        n_epochs=config.n_epochs or 10,
        n_steps=config.buffer_size or 2048,
        use_sde=config.use_sde or False,
        # tensorboard_log=config.get("log_dir", None),  # Uncomment if tensorboard log directory is specified
    )

    # Load pre-trained model if specified
    if config.model:
        print("Loading pretrained model from", config.model)
        model.set_parameters(os.path.join(config.model, "agent.zip"))

    return model


if __name__ == "__main__":
    # Set logging level for debug info and process visibility
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting training setup...")

    # Load environment and training configuration from YAML files
    env_config = load_config("env.yaml")
    train_config = load_config("train.yaml")

    # Create a unique folder to save training artifacts
    training_folder = create_training_folder(os.path.join(train_config["save_dir"], "training"), "sb3_ppo")
    train_config["save_dir"] = training_folder

    # Display loaded configuration for verification
    logging.info("Loaded environment configuration:")
    show_dict(env_config, depth=2)
    logging.info("Loaded training configuration:")
    show_dict(train_config, depth=2)

    # Save configurations into the designated training folder for reference
    save_config(training_folder, env_config, "env.yaml")
    save_config(training_folder, train_config, "train.yaml")

    # Initialize the environment and monitor it to log episode statistics
    env = BoosterEnv(config=EasyDict(env_config))
    env = Monitor(env, allow_early_resets=False)

    # Initialize the PPO model with training configuration and environment
    model = _initialize_ppo(config=EasyDict(train_config), env=env)

    # Calculate total training timesteps
    total_steps = train_config["episodes"] * env_config["max_steps"]

    # Set up a custom logger for tracking model progress
    new_logger = configure(training_folder, ["stdout", "csv", "tensorboard", "log"])
    model.set_logger(new_logger)

    # Start model training with callback for tracking metrics
    model.learn(
        total_timesteps=total_steps,
        callback=Sb3BoosterTracker(env_config, train_config)
    )
    logging.info("Training completed!")