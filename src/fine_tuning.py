import optuna
import numpy as np
from typing import Dict, Any
from easydict import EasyDict
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.ppo import PPO
from environment.env import BoosterEnv
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from training.booster_tracker import *
from common.yaml_helper import load_config
from common.utils import create_training_folder

# Constants for the optimization process
N_TRIALS = 50
N_STARTUP_TRIALS = 0
N_EVALUATIONS = 1
N_TIMESTEPS = int(100_000_000)  # Total timesteps for training
EVAL_FREQ = 3  # Evaluation frequency based on the number of timesteps
N_EVAL_EPISODES = 100  # Number of episodes for evaluation


def sample_reward_constants(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sample reward constants for the reinforcement learning environment.

    Args:
        trial (optuna.Trial): An instance of the current Optuna trial.

    Returns:
        Dict[str, Any]: A dictionary containing sampled reward constants.
    """
    # Define the possible values for each reward parameter
    space = [100, 300, 600, 900, 1200, 1500]
    R_s = trial.suggest_categorical("position", [25, 50, 75])
    R_vx = trial.suggest_categorical("vx", [40, 60, 80, 100])
    R_theta = trial.suggest_categorical("theta", [25, 50, 75, 100])
    dV = trial.suggest_categorical("dV", [-20, -10, 0, 10, 20])
    v = trial.suggest_categorical('V', [25, 50, 75, 100])

    return {
        "velocity": v,
        'Vx': R_vx,
        'Vy': R_vx + dV,
        "position": R_s,
        "angle": R_theta,
        "w": 0,
        "fuel_penalization": 0,
        "time_penalization": 0,
        "termination_reward": 30,
        "main_engine_burn": 0.01,
        "side_engine_burn": 0.005,
        "engine_startup": 0,
        "gimbal": 0.01,
    }


def objective(trial: optuna.Trial) -> float:
    """
    Objective function to optimize the RL model using Optuna.

    Args:
        trial (optuna.Trial): An instance of the current Optuna trial.

    Returns:
        float: The mean reward achieved in the trial.
    """
    # Load configuration for the environment from a YAML file
    kwargs = load_config('env.yaml')
    kwargs["reward"] = sample_reward_constants(trial)  # Sample reward constants for this trial
    kwargs["reward_version"] = trial.suggest_categorical('version', choices=['v6', 'v7'])

    # Initialize the PPO model
    model = PPO(
        "MlpPolicy",
        BoosterEnv(config=EasyDict(kwargs), render=False),
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_rate=get_linear_fn(0.001, 0.0003, 0.01),
        gamma=0.995,
    )

    # Create a callback for evaluating the success rate of the trial
    eval_callback = SuccessRateTrialCallback(
        EasyDict(kwargs),
        trial,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=0,
        max_iter=500
    )

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)  # Train the model
        # Record various attributes from the evaluation
        trial.set_user_attr('success_rate', eval_callback.success_rate)
        trial.set_user_attr('best_terminal_vx', eval_callback.best_ep.state_history[-2]["Vx"])
        trial.set_user_attr('best_terminal_vy', eval_callback.best_ep.state_history[-2]["Vy"])
        trial.set_user_attr('best_terminal_alpha', eval_callback.best_ep.state_history[-2]["alpha"])
        trial.set_user_attr('best_terminal_distance', np.linalg.norm((
            eval_callback.best_ep.state_history[-2]["x"],
            eval_callback.best_ep.state_history[-2]["y"]
        )))
    except AssertionError as e:
        # Handle cases where NaN values are encountered during training
        print(e)
        nan_encountered = True
    finally:
        # Ensure the environment is properly closed to free up resources
        model.env.close()

    # If NaN values were encountered, signal that the trial has failed
    if nan_encountered:
        return float("nan")

    # Check if the trial should be pruned based on the evaluation callback
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.mean_rew  # Return the mean reward for this trial


def study_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    """
    Callback function to save the study results after optimization.

    Args:
        study (optuna.Study): The study instance.
        trial (optuna.trial.FrozenTrial): The current trial instance.
    """
    study.trials_dataframe().sort_values(by="value").to_csv(f"{path}/study.csv")


if __name__ == "__main__":
    # Create a directory for saving training results
    path = create_training_folder('optimize_studies', 'rew')

    # Define the Optuna sampler and pruner
    sampler = TPESampler()  # Tree-structured Parzen Estimator sampler
    pruner = MedianPruner()  # Pruner to stop trials that are not promising

    # Create a study to optimize the objective function
    study = optuna.create_study(direction="maximize", pruner=pruner)

    try:
        # Optimize the study using the objective function
        study.optimize(objective, n_trials=N_TRIALS, timeout=None, n_jobs=1, callbacks=[study_callback])
    except KeyboardInterrupt:
        # Handle interruption gracefully
        pass
