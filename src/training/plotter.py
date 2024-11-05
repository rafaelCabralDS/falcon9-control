import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from environment.constants import (K_time, RENDER_FPS, LAUNCH_PAD_RADIUS,GROUND_HEIGHT, LAUNCH_PAD_HEIGHT)
import warnings
from typing import List
import os

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")


# Utility function to generate time steps
def get_time_steps(length):
    return [((K_time * t) / RENDER_FPS) for t in range(length)]


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def plot_iterations(save_dir, episodic_returns, episodic_lengths):
    if len(episodic_returns) != len(episodic_lengths):
        raise ValueError("Length mismatch between episodic returns and lengths")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    iterations = np.arange(len(episodic_returns))

    sns.lineplot(x=iterations, y=episodic_returns, ax=ax1, label="Episodic Return", color='b')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Episodic Return", color='b')

    ax2 = ax1.twinx()
    sns.lineplot(x=iterations, y=episodic_lengths, ax=ax2, label="Episodic Length", color='r')
    ax2.set_ylabel("Episodic Length", color='r')

    save_plot(fig, f"{save_dir}/training.png")


def plot_loss(save_dir, loss):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=np.arange(len(loss)), y=loss, ax=ax, color="purple")
    ax.set_title("Agent Loss Over Time")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")

    save_plot(fig, f"{save_dir}/agent_loss.png")

def plot_episodes_rewards(save_dir, rewards):
    episodes = np.arange(len(rewards))
    sns.scatterplot(x=episodes, y=rewards)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(f"{save_dir}/rewards.png")
    plt.close()


def plot_terminations(save_dir, terminations: List[str]):
    palette_color = sns.color_palette('hls', 8)

    # plotting data on chart
    plt.title("Terminations")
    categories, counts = np.unique(terminations, return_counts=True)
    plt.pie(counts, labels=categories, colors=palette_color,
            # explode=explode,
            autopct='%.0f%%')
    plt.savefig(os.path.join(save_dir, "terminations"))
    plt.close()

def plot_rewards(save_dir, rewards):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=np.arange(len(rewards)), y=rewards, ax=ax, color="green")
    ax.set_title("Rewards Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")

    save_plot(fig, f"{save_dir}/rewards.png")


def plot_pitch_angle(save_dir, angle, angular_velocity):
    if not angle or not angular_velocity:
        print("Warning: Empty angle or angular_velocity data")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    time_steps = get_time_steps(len(angle))

    sns.lineplot(x=time_steps, y=np.rad2deg(angle), ax=ax1, color='blue', label="Pitch Angle (Theta)")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Pitch Angle [deg]", color='blue')

    ax2 = ax1.twinx()
    sns.lineplot(x=time_steps, y=angular_velocity, ax=ax2, color='orange', label="Angular Velocity (Theta Dot)")
    ax2.set_ylabel("Angular Velocity [rad/s]", color='orange')

    fig.suptitle("Pitch Angle and Angular Velocity Over Time")
    save_plot(fig, f"{save_dir}/pitch_angle.png")


def plot_control(save_dir, main_engine, side_engine, nozzle_angle):
    if not main_engine or not side_engine or not nozzle_angle:
        print("Warning: Empty control data")
        return

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Agent Control Signals")

    t = get_time_steps(len(main_engine))

    # Main Engine Power
    axs[0].plot(t, [max(p, 0) for p in main_engine], label="Main Engine Power", color="blue")
    axs[0].set_title("Main Engine Power")
    axs[0].grid(True)

    # Nozzle Angle
    nozzle_angle_deg = np.rad2deg(nozzle_angle)
    axs[1].plot(t, nozzle_angle_deg, color="purple")
    axs[1].set_title("Nozzle Angle [deg]")
    axs[1].grid(True)

    # Side Engine
    side_active = [(time, power) for time, power in zip(t, side_engine) if abs(power) > 0.5]
    if side_active:
        times, powers = zip(*side_active)
        axs[2].scatter(times, powers, color="green")
    axs[2].set_ylim(-1.1, 1.1)
    axs[2].set_title("Side Engine Activation")
    axs[2].grid(True)

    save_plot(fig, f"{save_dir}/agent_control.png")


def plot_velocity(save_dir, Vx=None, Vy=None):
    if Vx is None and Vy is None:
        print("Warning: No velocity data provided")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    time_steps = get_time_steps(len(Vx or Vy))

    if Vx is not None:
        sns.lineplot(x=time_steps, y=Vx, ax=ax, label="Vx", color="blue")
    if Vy is not None:
        sns.lineplot(x=time_steps, y=Vy, ax=ax, label="Vy", color="green")

    ax.set_title("Velocity Profile")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [m/s]")

    save_plot(fig, f"{save_dir}/velocity_profile.png")


def plot_trajectory(save_dir, x, y):
    if not x or not y:
        print("Warning: Empty trajectory data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, ax=ax, marker='o', color="blue", label="Trajectory")
    ax.scatter([-LAUNCH_PAD_RADIUS, LAUNCH_PAD_RADIUS], [GROUND_HEIGHT + LAUNCH_PAD_HEIGHT] * 2,
               color='red', marker='x', label="Launchpad")
    ax.scatter(x[0], y[0], color='b', marker='o', label="Initial Position")
    ax.legend()
    ax.set(
        xlabel="x [m]",
        ylabel="y [m]",
        title="Trajectory",
        xlim=(-max([*x, abs(min(x))]) * 1.2, max([*x, abs(min(x))]) * 1.2),
        ylim=(0, max(y) * 1.2)
    )

    save_plot(fig, f"{save_dir}/trajectory_profile.png")


def plot_mission(save_dir, X, V, THETA):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Trajectory
    x, y = X
    sns.scatterplot(x=x, y=y, ax=axs[0], color="blue", label="Trajectory")
    axs[0].scatter([-LAUNCH_PAD_RADIUS, LAUNCH_PAD_RADIUS], [GROUND_HEIGHT + LAUNCH_PAD_HEIGHT] * 2,
                   color='red', marker='x', label="Launchpad")
    axs[0].scatter(x[0], y[0], color='b', marker='o', label="Initial Position")
    axs[0].legend()
    axs[0].set(title="Trajectory", xlabel="x [m]", ylabel="y [m]")

    # Velocity Profile
    Vx, Vy = V
    time_steps = get_time_steps(len(Vx))
    sns.lineplot(x=time_steps, y=Vx, ax=axs[1], label="Vx", color="blue")
    sns.lineplot(x=time_steps, y=Vy, ax=axs[1], label="Vy", color="green")
    axs[1].set(title="Velocity Profile", xlabel="Time [s]", ylabel="Velocity [m/s]")

    # Pitch Angle
    sns.lineplot(x=time_steps, y=np.rad2deg(THETA), ax=axs[2], color="purple", label="Pitch Angle (Theta)")
    axs[2].set(title="Pitch Angle", xlabel="Time [s]", ylabel="Angle [deg]")

    save_plot(fig, f"{save_dir}/mission.png")
