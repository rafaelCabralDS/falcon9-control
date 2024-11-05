import numpy as np
from enum import Enum

# --- General Simulation Constants ---
# Note: These constants have been empirically tested and should not be changed.
RENDER_FPS = 240
SI_UNITS_SCALE = 100  # Scale factor to convert simulation units to SI units
dt = 1 / RENDER_FPS   # Time step in seconds
K_time = 10           # Time scale constant for Box2D to SI units
K_velocity = SI_UNITS_SCALE / K_time  # Velocity scale constant
K_w = 0.04293         # Angular velocity scale constant
K_angle = 1           # Angle scale constant

# --- State Variables ---
class State(Enum):
    X_POSITION = 0
    Y_POSITION = 1
    X_VELOCITY = 2
    Y_VELOCITY = 3
    ANGLE = 4
    ANGULAR_VELOCITY = 5
    FUEL = 6
    PREVIOUS_ENGINE_POWER = 7

# State indices for easy access in arrays
XX = State.X_POSITION.value
YY = State.Y_POSITION.value
X_DOT = State.X_VELOCITY.value
Y_DOT = State.Y_VELOCITY.value
ALPHA = State.ANGLE.value
ALPHA_DOT = State.ANGULAR_VELOCITY.value
FUEL = State.FUEL.value
PREV_MAIN_POWER = State.PREVIOUS_ENGINE_POWER.value

# --- Booster Specifications ---
FIRST_STAGE_FUEL_CAPACITY = 395600  # kg, fuel capacity for RP-1 + LOX
BOOSTER_EMPTY_MASS = 25600          # kg, empty mass of the booster
BOOSTER_RADIUS = 1.85               # m, radius of the booster
BOOSTER_HEIGHT = 41.2               # m, height of the booster

# --- Nozzle and Main Engine Constants ---
NOZZLE_RADIUS = 0.46                # m, radius of the main nozzle
NOZZLE_AREA = 0.9                   # m^2, area of the main nozzle
MAX_GIMBAL_ANGLE = 5                # degrees, max gimbal angle for engine control
MAX_GIMBAL_ANGLE_RAD = np.deg2rad(MAX_GIMBAL_ANGLE)  # Convert to radians
MAX_GIMBAL_VELOCITY = 10            # degrees/s, max gimbal angle change rate

# Merlin-1D Main Engine Performance Constants
M1D_OXIDIZER_FUEL_RATIO = 2.38      # LOX/RP-1 mixture ratio (phi)
N_ENGINES = 3                       # Number of engines
M1D_MAX_THRUST = 845000             # N, max thrust per engine
M1D_THRESHOLD = 0.57                # Min throttle threshold
M1D_MIN_THRUST = M1D_MAX_THRUST * M1D_THRESHOLD  # N, minimum thrust at throttle threshold
M1D_VELOCITY_EXHAUST = 803.7        # m/s, exhaust velocity
M1D_EXIT_PRESSURE = 65400           # Pa, exhaust pressure at nozzle exit
M1D_SPECIFIC_IMPULSE = 304          # sec, specific impulse (vacuum)

# --- Side Engines (Draco) ---
DRACO_THRUST = 25000                # N, thrust per Draco engine

# --- Environmental Constants ---
EARTH_GRAVITY = 9.81                # m/s^2, gravitational acceleration
ATM_PRESSURE = 10e5                 # Pa, atmospheric pressure

# --- Landing Constraints ---
TILT_ANGLE = np.deg2rad(60)             # rad, max tilt angle during descent
TILT_LAND_ANGLE = np.deg2rad(23)        # rad, max tilt angle allowed for stable landing
ANGULAR_VELOCITY_LIMIT = 10             # rad/s, max allowed angular velocity
ANGULAR_VELOCITY_LAND_LIMIT = 1         # rad/s, safe angular velocity for landing
Y_LAND_VELOCITY_LIMIT = 10              # m/s, max vertical landing speed
X_LAND_VELOCITY_LIMIT = 2               # m/s, max horizontal landing speed

# --- Launch Pad Specifications ---
GROUND_HEIGHT = 10                    # m, height of the ground reference
LAUNCH_PAD_CENTER = 5000 / 2          # m, center position of the launch pad
LAUNCH_PAD_HEIGHT = 2                 # m, height of the launch pad
LAUNCH_PAD_RADIUS = 85 / 2            # m, radius of the launch pad

# --- Simulation Boundaries ---
X_LIMIT = 5000                        # m, x-axis boundary for simulation
Y_LIMIT = 5000                        # m, y-axis boundary for simulation

# --- Render Settings ---
VIEWPORT_SIZE = 600                   # Render viewport size in pixels
PIXELS_UNITS_SCALE = 80               # Scale factor for pixels to simulation units
