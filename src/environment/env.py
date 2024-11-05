from typing import Optional, Tuple, List, Any, Union
import numpy as np
from gymnasium import Env, spaces
from gymnasium.core import RenderFrame, ActType
from common.yaml_helper import load_config
from .utils import noisy
from .painter import TrackPainter
from .rewards import Reward
from .world import World
from easydict import EasyDict
from .constants import *

class BoosterEnv(Env):
    """
    A reinforcement learning environment for simulating the Falcon 9 landing problem environment.

    Action Space:
        0. Main engine: throttle control from -1 (off) to +1 (100% power).
        1. Side engine: left (-1 to -0.5), off (-0.5 to 0.5), right (0.5 to 1).
        2. Delta gimbal angle: tilt control from -1 to +1.

    Observation Space:
        0. Relative X position to the launchpad [m], within [-X_LIM, +X_LIM].
        1. Relative Y position to the launchpad [m], within [GROUND_HEIGHT, +Y_LIM].
        2. Horizontal velocity Vx [m/s], unbounded.
        3. Vertical velocity Vy [m/s], unbounded.
        4. Pitch angle [rad].
        5. Angular velocity w [rad/s].
        6. Fuel level relative to initial capacity [0-1].
        7. Previous main engine power [0-1].
        8. Previous nozzle angle [rad].
    """

    def __init__(self, config: Union[EasyDict, str, dict], render: Optional[bool] = False):
        super().__init__()

        self._loadEnvConfig(config)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.render_mode = "human" if (self.config["render"] or render) else "rgb_array"
        self.painter = TrackPainter(render_mode=self.render_mode)
        self.world: World = None

        # Initialize internal state variables
        self.initial_state = None
        self.previousState = None
        self.currentAction = None
        self.previousAction = None
        self._is_env_ready = False
        self.termination_cause = None
        self.reward = Reward(self.config)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment to an initial state.

        This function initializes the world, sets the initial state with noise, and prepares
        the environment for a new episode of training.

        Returns:
            Tuple: The initial observation of the environment and an empty dictionary.
        """
        super().reset(seed=seed)

        if self.world is not None:
            self.world.destroy()

        # Add initial state noise
        self.initial_state = EasyDict(dict(self.config.initial_condition))
        for key, value in dict(self.config.initial_condition).items():
            mean, sigma = [float(v) for v in value.strip().split("+-")]
            if sigma == 0:
                self.initial_state[key] = mean
            else:
                self.initial_state[key] = np.clip(noisy(mean, sigma), mean - sigma, mean + sigma)

        # Random direction
        initial_direction = [-1, 1][np.random.randint(0, 2)]
        self.initial_state["Vx"] *= -initial_direction
        self.initial_state["alpha"] *= initial_direction
        self.initial_state["x"] = LAUNCH_PAD_CENTER + (
                    initial_direction * (LAUNCH_PAD_CENTER - self.initial_state["x"]))

        self.world = World(
            np_random=self.np_random,
            initial_state=self.initial_state,
            wind_power=self.config.wind,
            turbulence=self.config.turbulence,
            drag=self.config.drag,
        )

        # Set rewards helper class
        self.reward.reset(
            X0=self.world.xy_target_pos,
            V0=(self.world.state.Vx, self.world.state.Vy)
        )

        # Initialize actions and state records
        self.previousAction = [0, 0, 0]
        self.currentAction = [0, 0, 0]
        self._is_env_ready = True
        self.previousState = self.state
        self.termination_cause = None

        return np.array(self.state, dtype=np.float32), {}

    @property
    def state(self) -> np.ndarray:
        """Returns the current state of the environment as an observation array."""
        state = self.world.state
        expanded_state = (
            self.world.xy_target_pos[0],
            self.world.xy_target_pos[1],  # Relative to target
            state.Vx,
            state.Vy,
            state.angle,
            state.w,
            state.fuel / FIRST_STAGE_FUEL_CAPACITY,
            self.previousAction[0],
            state.nozzle_angle,
        )
        return np.array(expanded_state, dtype=np.float32)

    @property
    def obs(self):
        """Other environment useful variables + latest action"""
        return {
            "t": self.world.state.t,
            "termination_cause": self.termination_cause,
            "drag": self.world.state.drag,
            "turbulence": self.world.state.turbulence,
            "wind": self.world.state.wind,
            "action": self.currentAction,
            "F": self.world.state.F,
            "nozzle_angle": self.world.state.nozzle_angle,
            "state": self.state,
        }

    def step(self, action: ActType) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Takes an action in the environment and returns the new state, reward,
        termination status, and additional information.

        Parameters:
            action (ActType): The action to take, represented as an array of floats.

        Returns:
            state\ (ObsType): The new booster state
            reward (float): The agent´s reward
            terminated (bool): A flag indicating if the episode is done
            truncated (bool): A flag indicating if the max episode steps have being reached
            obs (dict): A dict containing util informations about the entire environment state

        """
        if not self._is_env_ready:
            raise Exception("Environment is not ready. Call reset() first.")

        # Step the world forward with the given action
        self.world.step(action)

        # Evaluate termination conditions
        terminated, self.termination_cause = self._eval_termination()

        # Update the records
        self.previousAction = self.currentAction
        self.currentAction = action
        self.previousState = self.state

        # Calculate rewards based on the current state and action taken
        self.reward.calculate(
            terminated_successfully=terminated,
            X=(self.state[0], self.state[1]),  # Current X and Y positions
            V=(self.state[2], self.state[3]),  # Current velocities
            alpha=self.state[4],  # Current angle
            w=self.state[5],  # Current angular velocity
            action=action,  # Current action
            action_prev=self.previousAction,  # Previous action
            previous_state=self.previousState,  # Previous state
            step=self.world.state.t / (dt * K_time),  # Time step
            fuel=self.world.state.fuel,  # Current fuel level
        )

        self.render()  # Render the current state if applicable

        return self.state, self.reward.current, terminated is not None, False, self.obs

    def _eval_termination(self):
        """
        Evaluates the termination conditions for the environment.

        Returns:
            Tuple: A tuple containing a boolean indicating if the episode has terminated,
            and a string describing the reason for termination.
        """
        # Time Limit
        step = self.world.state.t / (dt * K_time)
        if self.config.max_steps is not None and step > self.config.max_steps:
            return False, "Time Limit"

        # If the booster touched the ground, check if the conditions configure a successfully landing
        if self.world.contact:
            exploded = any([
                abs(self.previousState[3]) >= Y_LAND_VELOCITY_LIMIT,  # Y velocity limit
                abs(self.previousState[2]) >= X_LAND_VELOCITY_LIMIT,  # X velocity limit
                abs(self.previousState[4]) >= TILT_LAND_ANGLE,  # Pitch angle limit
                abs(self.previousState[0]) > LAUNCH_PAD_RADIUS,  # Outside launchpad radius
            ])

            if exploded:
                return False, "Explosion"
            else:
                return True, "Landed"

        # Check for stability violations
        if abs(self.state[0]) > X_LIMIT: return False, "X limit"
        if abs(self.state[1]) > Y_LIMIT: return False, "Y limit"
        if abs(self.state[4]) > TILT_ANGLE: return False, "Tilted"
        if abs(self.state[5]) > ANGULAR_VELOCITY_LIMIT: return False, "W"

        return None, None  # No termination condition met, keep going

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """Renders the current state of the environment if rendering mode is active."""
        if self.render_mode == "human":
            self.painter.paint(world=self.world)
        return

    def close(self):
        """Closes the environment and cleans up resources."""
        if self.painter is not None:
            self.painter.dispose()
        if self.world is not None:
            self.world.destroy()

    def _loadEnvConfig(self, config: Union[EasyDict, str, dict]):
        """Loads the environment configuration from a given source."""
        if isinstance(config, EasyDict):
            self.config = config
        elif isinstance(config, str):
            self.config = EasyDict(load_config(config))
        elif isinstance(config, dict):
            self.config = EasyDict(config)
        else:
            raise Exception("Could not read the config file")
