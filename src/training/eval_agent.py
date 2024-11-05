import os
import time
from typing import Union, List, Dict, Tuple
from easydict import EasyDict
from environment.env import BoosterEnv
from .episode_recorder import BoosterRecorder
from .plotter import plot_episodes_rewards, plot_terminations
from stable_baselines3.common.policies import BasePolicy
from gymnasium import Env
import numpy as np


def eval_sb3_agent(
        agent: BasePolicy,
        save_dir: str,
        env_: Union[EasyDict, Env, str, Dict],
        render: bool = True,
        n: int = 30,
        verbose: bool = True
) -> Tuple[float, float, float]:
    """
    Evaluate a Stable Baselines3 agent in a specified environment.

    Args:
        agent (BasePolicy): The agent to be evaluated.
        save_dir (str): Directory where evaluation results will be saved.
        env_ (Union[EasyDict, Env, str, Dict]): The environment in which the agent will be evaluated.
        render (bool): Whether to render the environment during evaluation. Default is True.
        n (int): Number of episodes to evaluate. Default is 30.
        verbose (bool): Whether to print progress information. Default is True.

    Returns:
        Tuple[float, float, float]: A tuple containing the success rate, mean total reward, and standard deviation of total rewards.
    """

    episodes: List[BoosterRecorder] = []
    best_episode: BoosterRecorder = None

    # Initialize environment based on the type of env_
    if isinstance(env_, Env):
        env = env_
    else:
        env_config = EasyDict(env_) if isinstance(env_, (EasyDict, str, dict)) else None
        env = BoosterEnv(env_config, render=render) if env_config else None

    if env is None:
        raise ValueError("Invalid environment provided. Must be an instance of EasyDict, Env, str, or dict.")

    for i in range(n):
        # Initialize new episode
        state, _ = env.reset()
        episode_recorder = BoosterRecorder()
        episodes.append(episode_recorder)

        while not episode_recorder.done:
            action, _ = agent.predict(state)
            state, reward, done, _, obs = env.step(action)
            episode_recorder.on_step(state, action, reward, done, obs)

            if render:
                time.sleep(1 / 60)  # Simulate a frame rate

        episode_recorder.terminate(obs.get("termination_cause", "Unknown"))

        # Save "best" episode based on total reward
        if best_episode is None or episode_recorder.total_reward > best_episode.total_reward:
            best_episode = episode_recorder

        if verbose:
            success_count = len([ep for ep in episodes if ep.success])
            success_rate = (success_count / (i + 1)) * 100
            print(f"Successfully landed: {success_count} | Success Rate: {success_rate:.2f}%\r", flush=True)

    if save_dir:
        eval_dir = os.path.join(save_dir, "eval")
        best_dir = os.path.join(eval_dir, "best")
        os.makedirs(eval_dir, exist_ok=True)
        os.makedirs(best_dir, exist_ok=True)

        # Plotting results
        plot_episodes_rewards(eval_dir, [ep.total_reward for ep in episodes])
        plot_terminations(eval_dir, [ep.termination_reason for ep in episodes])
        best_episode.plot(best_dir)
        best_episode.to_csv(os.path.join(best_dir, "episode.csv"))

    # Calculate success rate
    termination_reasons = [ep.termination_reason for ep in episodes]
    categories, counts = np.unique(termination_reasons, return_counts=True)

    success_rate = counts[categories.tolist().index('Landed')] if 'Landed' in categories else 0.0

    # Close the environment
    try:
        env.close()
    except Exception as e:
        print(f"Error closing the environment: {e}")

    # Calculate mean and standard deviation of total rewards
    total_rewards = [ep.total_reward for ep in episodes]
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    return success_rate, mean_reward, std_reward
