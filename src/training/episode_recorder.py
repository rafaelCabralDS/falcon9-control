from .plotter import *
import pandas as pd

from environment.constants import XX, YY, X_DOT, Y_DOT, ALPHA, ALPHA_DOT, FUEL


class BoosterRecorder:
    """
    A class for recording, managing, and visualizing an episode's data during reinforcement learning in a booster landing environment.
    Tracks states, actions, rewards, and provides utilities for data extraction, CSV export, and visualization.
    """

    def __init__(self):
        self._actions_history = []
        self._state_history = []
        self.episode_rewards = []
        self.episode_steps = 0
        self.done = False
        self.obs_history = []
        self.termination_reason = None

    def __getitem__(self, i):
        """
        Allows indexing into the BoosterRecorder as if it were a DataFrame.

        Args:
            i (int): Index of the desired step data.

        Returns:
            pd.Series: The data at the specified index in DataFrame format.
        """
        return self.to_frame().iloc[i]

    def on_step(self, state, action, reward, done, obs):
        """
        Records a single step of the episode.

        Args:
            state (dict): Current state of the environment.
            action (tuple): Action taken at the current step.
            reward (float): Reward received after taking the action.
            done (bool): Whether the episode has ended.
            obs (dict): Additional observations at the current step.
        """
        if self.done:
            raise RuntimeError("Attempted to call on_step after episode has finished.")

        self.episode_steps += 1
        self._state_history.append(state)
        self._actions_history.append(action)
        self.episode_rewards.append(reward)
        self.obs_history.append(obs)
        self.done = done

        if done:
            self.terminate(obs.get("termination_cause", "Unknown"))

    def terminate(self, termination_reason):
        """
        Marks the episode as finished and records the termination reason.

        Args:
            termination_reason (str): The reason for episode termination.
        """
        self.done = True
        self.termination_reason = termination_reason

    @property
    def success(self):
        """bool: True if the episode terminated with a successful landing."""
        return self.termination_reason == "Landed"

    @property
    def total_reward(self):
        """float: The total reward accumulated over the episode."""
        return sum(self.episode_rewards)

    @property
    def mean_reward(self):
        """float: The average reward per step over the episode."""
        return self.total_reward / self.episode_steps if self.episode_steps > 0 else 0

    def to_csv(self, filename: str):
        """
        Exports the recorded episode data to a CSV file.

        Args:
            filename (str): Path to the output CSV file.
        """
        self.to_frame().to_csv(filename, index=False)

    def to_frame(self) -> pd.DataFrame:
        """
        Converts the recorded episode data into a DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the state, action, and reward history.
        """
        episode_data = pd.concat([
            pd.DataFrame(self.state_history),
            pd.DataFrame(self.actions_history),
            pd.Series(self.episode_rewards, name="reward"),
            pd.Series([None] * (self.episode_steps - 1) + [self.termination_reason], name="termination_reason")
        ], axis=1)

        return episode_data

    @property
    def state_history(self):
        """list of dict: The history of states recorded during the episode."""
        return [
            {
                "x": state[XX],
                "y": state[YY],
                "Vx": state[X_DOT],
                "Vy": state[Y_DOT],
                "alpha": state[ALPHA],
                "w": state[ALPHA_DOT],
                "fuel": state[FUEL],
            }
            for state in self._state_history
        ]

    @property
    def actions_history(self):
        """list of dict: The history of actions recorded during the episode."""
        return [
            {
                "main_engine": action[0],
                "side_engine": action[1],
                "gimbal": action[2],
            }
            for action in self._actions_history
        ]

    def get_att_history(self, att: str):
        """
        Retrieves the history of a specific attribute from the state data.

        Args:
            att (str): The attribute name to retrieve.

        Returns:
            list: The list of values for the specified attribute.
        """
        return [state[att] for state in self.state_history]

    def get_action_att_history(self, att: str):
        """
        Retrieves the history of a specific attribute from the action data.

        Args:
            att (str): The attribute name to retrieve.

        Returns:
            list: The list of values for the specified action attribute.
        """
        return [action[att] for action in self.actions_history]

    def get_obs_att_history(self, att: str):
        """
        Retrieves the history of a specific attribute from the observation data.

        Args:
            att (str): The attribute name to retrieve.

        Returns:
            list: The list of values for the specified observation attribute.
        """
        return [obs[att] for obs in self.obs_history]

    def plot(self, save_dir):
        """
        Generates plots for mission trajectory, velocity, pitch angle, rewards, and control inputs.

        Args:
            save_dir (str): Directory to save the plots.
        """
        plot_mission(
            save_dir,
            X=(self.get_att_history("x"), self.get_att_history("y")),
            V=(self.get_att_history("Vy"), self.get_att_history("Vx")),
            THETA=self.get_att_history("alpha")
        )

        plot_velocity(save_dir, Vy=self.get_att_history("Vy"), Vx=self.get_att_history("Vx"))
        plot_pitch_angle(save_dir, self.get_att_history("alpha"), self.get_att_history("w"))
        plot_trajectory(save_dir, self.get_att_history("x"), self.get_att_history("y"))
        plot_rewards(save_dir, self.episode_rewards)
        plot_control(
            save_dir,
            self.get_action_att_history("main_engine"),
            self.get_action_att_history("side_engine"),
            self.get_obs_att_history("nozzle_angle")
        )
