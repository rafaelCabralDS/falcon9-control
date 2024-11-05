import abc
import logging
import os
#import torch
import numpy as np
import pandas as pd
import gymnasium as gym
import optuna
from abc import ABC
from typing import Union, Optional, Dict, Any
from .episode_recorder import BoosterRecorder
from common.utils import clear_folder
from .plotter import *
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.policies import BasePolicy
from easydict import EasyDict
from .eval_agent import eval_sb3_agent
#from booster_env.env import BoosterEnv


class Callback(ABC):

    """
        Abstract base class for a callback that tracks training progress and performance in reinforcement learning (RL).

    Attributes:
        env_config (Dict): Environment configuration dictionary.
        train_config (Dict): Training configuration dictionary, including parameters for evaluation and saving models.
    """

    def __init__(self, env_config: Dict, train_config: Dict):
        self._log_df = pd.DataFrame()
        self.env_config = env_config
        self.save_dir = train_config.get("save_dir", ".")
        self._max_iter = train_config.get("max_iter", np.inf)
        self._save_freq = train_config.get("save_freq", 10)
        self._eval_freq = train_config.get("eval_freq", 50)

        self.num_timesteps = 0
        self.episodes = 0
        self.episode = BoosterRecorder()
        self.max_reward = -np.inf
        self.episodes_reward = []
        self.termination_history = []
        self._rollouts = 0

    def _on_step(self, state, reward, done, obs, actions) -> bool:
        """
        Updates the callback on each step taken by the agent.

        Args:
            state: Current state of the environment.
            reward: Reward obtained after taking the action.
            done: Boolean indicating if the episode has ended.
            obs: Observation from the environment.
            actions: Actions taken by the agent.

        Returns:
            bool: Whether to continue training.
        """
        self.num_timesteps += 1
        self.episode.on_step(state, actions, reward, done, obs)

        if done:
            self.on_episode_end()
            self.on_episode_start()
        return True

    def on_episode_start(self):
        """Resets the episode recorder at the start of each new episode."""
        self.episodes += 1
        self.episode = BoosterRecorder()

    def on_episode_end(self):
        """Records and logs episode rewards and termination reasons. Saves best model based on episode reward"""
        self.episodes_reward.append(self.episode.total_reward)
        self.termination_history.append(self.episode.termination_reason)

        # Save best model
        if self.episode.total_reward > self.max_reward:
            self.max_reward = self.episode.total_reward
            best_dir = os.path.join(self.save_dir, "training_best_ep")
            clear_folder(best_dir)
            self.save_model(best_dir)
            self.episode.to_csv(os.path.join(best_dir, f"{self.episodes}.csv"))
            self.episode.plot(best_dir)

    def on_rollout_end(self) -> bool:
        """
        Executes evaluation, saving, and plotting logic at the end of each rollout.

        Returns:
            bool: Whether to continue training.
        """
        self._rollouts += 1
        self._read_log_file()

        # Eval
        if self._rollouts % self._eval_freq == 0:
            self._evaluate_model(self._rollouts)

        # Plot training
        if self._rollouts % self._save_freq == 0:
            self.save_model(self.save_dir)
            plot_iterations(self.save_dir, self._log_df["returns"], self._log_df["lengths"])
            if "loss" in self._log_df.columns.values:
                plot_loss(self.save_dir, self._log_df["loss"])

        if self._rollouts > self._max_iter:
            self.save_model(self.save_dir)
            plot_iterations(self.save_dir, self._log_df["returns"], self._log_df["lengths"])
            plot_loss(self.save_dir, self._log_df["loss"])
            print(f"Max update iterations of {self._max_iter} reached. Training is now done")
            return False
        return True

    def _evaluate_model(self, rollouts: int):
        eval_dir = os.path.join(self.save_dir, f"{self._rollouts}")
        os.makedirs(eval_dir)
        self.save_model(eval_dir)

        self.eval_model(
            agent=self.agent(),
            env_config=self.env_config,
            verbose=False,
            n=30,
            render=False,
            save_dir=eval_dir
        )

    @abc.abstractmethod
    def agent(self):
        pass

    @abc.abstractmethod
    def eval_model(self, agent, env_config, verbose, n, render, save_dir):
        pass

    @abc.abstractmethod
    def save_model(self, path):
        pass

    @abc.abstractmethod
    def _read_log_file(self):
        """Reads the training log file and updates the DataFrame for tracking rewards, lengths, and loss."""
        pass

class Sb3BoosterTracker(Callback, BaseCallback):
    """
    Callback for tracking the training progress of an SB3 (Stable Baselines3) model with periodic evaluation and saving.
    """

    def __init__(self, env_config: Dict, train_config: Dict):
        super().__init__(env_config, train_config)

    def on_step(self) -> bool:
        """
        Custom on_step function for Stable Baselines3, tracking episode progress, rewards, and actions.

        Returns:
            bool: Whether to continue training.
        """
        obs = self.locals['infos'][0]
        reward = self.locals['rewards'][0]
        done = self.locals["dones"][0]
        state = obs["state"]
        action = obs["action"]

        self._on_step(state, reward, done, obs, action)
        if self._rollouts > self._max_iter:
            print(f"Max iterations ({self._max_iter}) reached. Training complete.")
            return False
        return True

    def eval_model(self, agent, env_config, verbose, n, render, save_dir):
        eval_sb3_agent(agent, save_dir, env_config, render, n, verbose)

    def save_model(self, path):
        self.model.save(os.path.join(path, "agent.zip"))

    def agent(self):
        return self.model.policy

    def _read_log_file(self):

        # Read the csv file
        path = os.path.join(self.save_dir, "progress.csv")
        try:
            self._log_df = pd.read_csv(path)
            self._log_df = self._log_df.rename(columns={
                "rollout/ep_rew_mean": "returns",
                "rollout/ep_len_mean": "lengths",
                "train/loss": "loss",
            })
        except:  # Create a placeholder
            self._log_df = pd.DataFrame(columns=["returns", "lengths", "loss"])
            return



class SuccessRateTrialCallback(BaseCallback):
    """
    Callback for monitoring success rate and reward during hyperparameter tuning with Optuna.

    Attributes:
        eval_env (EasyDict | gym.Env): Evaluation environment.
        trial (optuna.Trial): Optuna trial object for reporting metrics.
        n_eval_episodes (int): Number of episodes for evaluation.
        eval_freq (int): Frequency of evaluation.
    """

    def __init__(
        self,
        eval_env: Union[EasyDict, gym.Env],
        trial: optuna.Trial,
        n_eval_episodes: int = 100,
        eval_freq: int = 10000,
        verbose: int = 0,
        max_iter: int = 5000
    ):
        super().__init__(verbose=verbose)
        self.trial = trial
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.max_iter = max_iter
        self._recorder = BoosterRecorder()
        self.iteration = 0
        self.eval_idx = 0
        self.success_rate = 0.0
        self.mean_rew = 0
        self.best_episode: Optional[BoosterRecorder] = None

    def _evaluate_and_report(self) -> bool:
        """
        Evaluates agent performance and reports results to Optuna trial for potential pruning.

        Returns:
            bool: Whether the trial should be pruned based on results.
        """
        self.success_rate, self.mean_rew, _ = eval_sb3_agent(
            self.model.policy,
            None,
            env_=self.eval_env,
            render=False,
            n=self.n_eval_episodes,
            verbose=False
        )
        logging.info(f"Eval mean reward: {self.mean_rew}, Success rate: {self.success_rate}")

        self.trial.report(self.mean_rew, self.eval_idx)
        if self.trial.should_prune():
            self.is_pruned = True
            return False
        return True

    def _on_step(self) -> bool:
        """
        Tracks agent training progress, triggering evaluation and reporting for Optuna if conditions are met.

        Returns:
            bool: Whether to continue training.
        """
        obs = self.locals['infos'][0]
        reward = self.locals['rewards'][0]
        done = self.locals["dones"][0]
        state = obs["state"]
        action = obs["action"]

        self._recorder.on_step(state, action, reward, done, obs)

        if done:
            if self.best_episode is None or self._recorder.total_reward > self.best_episode.total_reward:
                self.best_episode = self._recorder
            self._recorder = BoosterRecorder()  # Reset recorder after episode ends

        if self.iteration > self.max_iter:
            return False
        return True

