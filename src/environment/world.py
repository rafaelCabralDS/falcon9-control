from typing import Tuple
from Box2D import *
from constants import *
from utils import *
from easydict import EasyDict
from gymnasium.utils import seeding, EzPickle


class World(EzPickle):
    """
    World environment simulating a physical system using Box2D, with wind and turbulence and a very simple drag model.
    This class is designed to represent the dynamics of a booster in a physics-based simulation.

    Parameters:
    - np_random (np.random.Generator): Numpy random generator for stochastic elements.
    - initial_state (EasyDict): Booster initial state settings, including position, velocity, and angle.
    - wind_power (float): Magnitude of wind force to apply.
    - turbulence (float): Magnitude of turbulence force to apply.
    - drag (float): Fins dumping magnitude (Dumb drag model)
    """

    def __init__(self,
                 np_random: np.random.Generator,
                 initial_state: EasyDict,
                 wind_power: float = 5000.0,
                 turbulence: float = 5000.0,
                 drag=10000.0,
                 ):
        super().__init__(np_random, initial_state, wind_power, turbulence, drag)
        self.np_random = np_random
        self.initial_state = initial_state

        # Initialize the Box2D world with gravity
        self.world = Box2D.b2World(gravity=(0, -EARTH_GRAVITY))
        self.world._contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world._contactListener_keepref

        # Setup the simulation environment components
        self.terrain = self._createTerrain()
        self.launch_pad = self._createLaunchPad()
        self.booster = self._createBooster()
        self.particles = []

        # Step count and simulation parameters
        self._step = 0
        self._windPower = wind_power
        self._turbulencePower = turbulence
        self._drag = drag
        self._windStep = np.random.randint(-1000, 1000)
        self._turbulenceStep = np.random.randint(-1000, 1000)
        self.contact = False
        self.state = EasyDict()
        self._setInitialState(initial_state)

    @property
    def xy_target_pos(self) -> tuple[float, float]:
        """Calculate the distance to the target in the x-y plane."""
        booster_cg = (self.state.x, self.state.y)
        target = (LAUNCH_PAD_CENTER, LAUNCH_PAD_HEIGHT + GROUND_HEIGHT)
        return points_distance((self.state.x, self.state.y))[0]

    def step(self, action: np.ndarray) -> EasyDict:
        """
        Advance the simulation by one step.

        Parameters:
        - action (np.ndarray): Control inputs for main engine power and side thrusters.

        Returns:
        - EasyDict: Updated state information after the step.
        """
        self._step += 1
        self.contact = self.legs[0].ground_contact or self.legs[1].ground_contact
        action[0] = np.clip(action[0], -1, 1)  # Main engine power
        action[2] = np.clip(action[2], -1, 1)  # Gimbal adjustment

        # Control logic for engine firing based on contact status
        if not self.contact:
            self._fireMainEngine(action[0], action[2])
            self._fireSideThruster(action[1])

        # Apply environmental effects if above a certain altitude
        if self.state.y >= 1000:
            self._applyTurbulence()
            self._applyWind()

        if self._useDrag:
            self._applyFinsDamping()

        # Apply damping and update particles
        self._updateParticles()
        self.world.Step(dt, 6 * 60, 6 * 60)  # Run Box2D simulation step

        # Update the state variables
        self.set_state()
        return self.state

    def set_state(self):
        """Update the current state based on the simulation variables."""
        x, y = self.fuselage.worldCenter
        Vx, Vy = self.fuselage.linearVelocity
        self.state.t = self._step * dt * K_time
        self.state.x = x * SI_UNITS_SCALE
        self.state.y = y * SI_UNITS_SCALE
        self.state.angle = self.fuselage.angle * K_angle
        self.state.Vx = Vx * K_velocity
        self.state.Vy = Vy * K_velocity
        self.state.w = self.fuselage.angularVelocity * K_w
        self.state.legs_on_contact = sum(int(leg.ground_contact) for leg in self.legs)
        self.contact = self.state.legs_on_contact > 0

    def _setInitialState(self, state: EasyDict):
        """
        Set the initial position, velocity, angle, and fuel of the booster.

        Parameters:
        - state (EasyDict): Initial values for position, velocity, and fuel ratio.
        """
        # Convert metric values to simulation scale
        x = state.x / SI_UNITS_SCALE
        y = state.y / SI_UNITS_SCALE
        alpha = state.alpha / K_angle
        Vx = state.Vx / K_velocity
        Vy = state.Vy / K_velocity
        w = state.w / K_w
        fuel = state.fuel_ratio * FIRST_STAGE_FUEL_CAPACITY

        # Apply values to the simulation objects
        self.fuselage.position = (x, y)
        self.fuselage.linearVelocity = (Vx, Vy)
        self.fuselage.angularVelocity = w
        self.fuselage.angle = alpha
        self.state.fuel = fuel
        self.state.nozzle_angle = 0.0
        self.state.legs_on_contact = 0
        self.state.F = (0, 0)

        # Update state variables for consistency
        self.set_state()
        self._applyWind()
        self._applyTurbulence()

    # ------------------------------------- CONTROL ------------------------------------- #

    def _fireMainEngine(self, power: float, d_alpha: float):
        """
        Fire the main engine based on power and angle adjustments.

        Parameters:
        - power (float): Engine power level (normalized between 0 and 1).
        - d_alpha (float): Adjustment angle for nozzle gimbal.
        """
        if self.state.fuel <= 0 or power <= 0:
            return

        # Update nozzle angle within bounds
        self.state.nozzle_angle += np.deg2rad(MAX_GIMBAL_VELOCITY) * d_alpha * (dt * K_time)
        self.state.nozzle_angle = np.clip(self.state.nozzle_angle, -np.deg2rad(MAX_GIMBAL_ANGLE),
                                          np.deg2rad(MAX_GIMBAL_ANGLE))

        # Compute thrust direction and force components
        thrust_angle = self.fuselage.angle + self.state.nozzle_angle
        thrust_dispersion = self.np_random.uniform(-0.005, 0.005)
        thrust = N_ENGINES * M1D_MAX_THRUST * (M1D_THRESHOLD + (0.43 * power))
        thrust_x = thrust * (-np.sin(thrust_angle) + thrust_dispersion)
        thrust_y = thrust * (np.cos(thrust_angle) - thrust_dispersion)
        self.state.F = (thrust_x, abs(thrust_y))

        # Apply force and torque from thrust
        self.fuselage.ApplyForceToCenter((thrust_x, thrust_y), True)
        self.fuselage.ApplyTorque(thrust_x * ((BOOSTER_HEIGHT / 2) / SI_UNITS_SCALE), True)

        # Calculate fuel consumption
        consumed_fuel = N_ENGINES * (m_dot(
            F=thrust / N_ENGINES,
            Ve=M1D_Ve,
            Pe=M1D_Pe,
            _Pa=Pa,
            mixRatio=M1D_PHI,
            Ae=NOZZLE_AREA)[0] * dt * K_time)

        self.fuselage.mass -= consumed_fuel
        self.state.fuel -= consumed_fuel

    def _fireSideThruster(self, side_power: float):
        """
        Activate side thrusters for yaw control.

        Parameters:
        - side_power (float): Power for left [-1, 0.5[ or right ]0.5, 1] thruster activation.
        """
        side = np.sign(side_power)
        p = abs(side_power)
        if p < 0.5:
            return
        side_index = 0 if side == -1 else 1
        thruster = self.sideThrusters[side_index]

        # Calculate force direction and magnitude
        thrust_dispersion = self.np_random.uniform(-0.005, 0.005)
        thrust_y = DRACO_THRUST * p * (-np.sin(self.fuselage.angle) + thrust_dispersion)
        thrust_x = DRACO_THRUST * p * (np.cos(self.fuselage.angle) - thrust_dispersion)

        # Apply the force
        thruster.ApplyForce((-thrust_x, -thrust_y), thruster.worldCenter, True)

    def _applyWind(self):
        """Simulate wind effects by applying a horizontal force."""
        wind_mag = np.tanh(np.sin(0.02 * self._windStep) + np.sin(np.pi * 0.01 * self._windStep)) * self._windPower
        self._windStep += 1
        self.fuselage.ApplyForceToCenter((wind_mag, 0.0), True)

    def _applyTurbulence(self):
        """Simulate turbulence with rapid, stochastic force fluctuations."""
        turbulence_mag = np.tanh(
            np.sin(0.1 * self._turbulenceStep) + np.cos(np.pi * 0.3 * self._turbulenceStep)) * self._turbulencePower
        self._turbulenceStep += 1
        self.fuselage.ApplyForceToCenter((turbulence_mag, 0.0), True)

    def _applyFinsDamping(self):
        """Apply rotational damping using fins."""
        damping_torque = -self._drag * self.fuselage.angularVelocity
        self.fuselage.ApplyTorque(damping_torque, True)

    # --------------------------------------------- BUILD ------------------------------- #
    def _createTerrain(self) -> b2Body:
        terrain_coordinates = [(0, 0), (0, GROUND_HEIGHT), (X_LIMIT, GROUND_HEIGHT), (X_LIMIT, 0)]
        terrain_coordinates = [(x / SI_UNITS_SCALE, y / SI_UNITS_SCALE) for x, y in terrain_coordinates]
        terrain = self.world.CreateStaticBody(shapes=b2PolygonShape(vertices=terrain_coordinates))
        terrain.CreateEdgeFixture(
            vertices=terrain_coordinates,
            density=0,
            friction=1
        )
        return terrain

    def _createLaunchPad(self) -> b2Body:
        self.launchPadConstraints = EasyDict()

        self.launchPadConstraints.center = (LAUNCH_PAD_CENTER, GROUND_HEIGHT + LAUNCH_PAD_HEIGHT)
        self.launchPadConstraints._bottomLeft = (LAUNCH_PAD_CENTER - LAUNCH_PAD_RADIUS, GROUND_HEIGHT)
        self.launchPadConstraints._bottomRight = (LAUNCH_PAD_CENTER + LAUNCH_PAD_RADIUS, GROUND_HEIGHT)
        self.launchPadConstraints._topLeft = (LAUNCH_PAD_CENTER - LAUNCH_PAD_RADIUS, GROUND_HEIGHT + LAUNCH_PAD_HEIGHT)
        self.launchPadConstraints._topRight = (LAUNCH_PAD_CENTER + LAUNCH_PAD_RADIUS, GROUND_HEIGHT + LAUNCH_PAD_HEIGHT)

        pad_vertices = [(x / SI_UNITS_SCALE, y / SI_UNITS_SCALE) for x, y in [
            self.launchPadConstraints._bottomLeft, self.launchPadConstraints._topLeft,
            self.launchPadConstraints._topRight, self.launchPadConstraints._bottomRight,
        ]]

        launch_pad = self.world.CreateStaticBody(shapes=b2PolygonShape(vertices=pad_vertices))
        launch_pad.CreateEdgeFixture(
            vertices=pad_vertices,
            density=0,
            friction=3
        )
        return launch_pad

    def _createBooster(self):

        BOOSTER_POLY = [
            (-BOOSTER_RADIUS, 0), (+BOOSTER_RADIUS, 0),
            (+BOOSTER_RADIUS, +BOOSTER_HEIGHT), (-BOOSTER_RADIUS, +BOOSTER_HEIGHT)
        ]
        BOOSTER_POLY = [(x / SI_UNITS_SCALE, y / SI_UNITS_SCALE) for x, y in BOOSTER_POLY]

        LEGS_MASS = 2000
        INITIAL_MASS = (BOOSTER_EMPTY_MASS - LEGS_MASS) + (FIRST_STAGE_FUEL_CAPACITY * self.initial_state.fuel_ratio)
        FUSELAGE_DENSITY = INITIAL_MASS / (BOOSTER_HEIGHT * BOOSTER_RADIUS * 2) * (SI_UNITS_SCALE ** 2)
        initial_x, initial_y = self.initial_state.x / SI_UNITS_SCALE, self.initial_state.y / SI_UNITS_SCALE

        self.fuselage: Box2D.b2Body = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0,
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(vertices=BOOSTER_POLY),
                density=FUSELAGE_DENSITY,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0  # 0 (rigid) 0.99 (bouncy)
            ),
        )

        # ------------------------------------- LEGS ---------------------------------------------

        self.legs = []

        LEG_HEIGHT = 12  # BOOSTER_HEIGHT / 3
        LEG_WIDTH = 1
        LEG_ANGLE = np.deg2rad(45)

        for legDirection in [-1, 1]:

            leg_poly = [
                (0, 0),  # A
                (LEG_HEIGHT * np.sin(LEG_ANGLE) + LEG_WIDTH, LEG_HEIGHT * np.cos(LEG_ANGLE) + LEG_WIDTH),  # B
                (LEG_HEIGHT * np.sin(LEG_ANGLE) + LEG_WIDTH, LEG_HEIGHT * np.cos(LEG_ANGLE)),  # C
                (LEG_WIDTH, 0)  # D
            ]

            leg_poly = [(x / SI_UNITS_SCALE, y / SI_UNITS_SCALE) for x, y in leg_poly]

            leg: b2Body = self.world.CreateDynamicBody(
                position=(initial_x, initial_y),
                angle=np.deg2rad(270) if legDirection == 1 else 0,
                fixtures=b2FixtureDef(
                    shape=b2PolygonShape(vertices=leg_poly),
                    density=(2000 / (LEG_HEIGHT * LEG_WIDTH) * (SI_UNITS_SCALE ** 2)),
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x005
                )
            )
            leg.ground_contact = False

            joint = b2WeldJointDef(
                bodyA=self.fuselage,
                bodyB=leg,
                localAnchorA=(legDirection * BOOSTER_RADIUS / SI_UNITS_SCALE, 5.2 / SI_UNITS_SCALE),
                localAnchorB=leg_poly[1] if legDirection == -1 else leg_poly[0],
            )

            leg.joint = self.world.CreateJoint(joint)

            self.legs.append(leg)

            # ---------------------------------- SIDE BOOSTERS ------------------------------- #
            self.sideThrusters = []
            for side in [-1, 1]:
                w, h = 0.6, 1.2
                thruster = self.world.CreateDynamicBody(
                    position=(initial_x, initial_y),
                    angle=0,
                    fixtures=b2FixtureDef(
                        shape=b2PolygonShape(box=(w / SI_UNITS_SCALE, h / SI_UNITS_SCALE)),
                        density=FUSELAGE_DENSITY,
                        friction=0.1,
                        categoryBits=0x0050,
                        maskBits=0x001,  # collide only with ground
                        restitution=0.0  # 0 (rigid) 0.99 (bouncy)
                    ),
                )
                thruster.joint = self.world.CreateJoint(
                    b2WeldJointDef(
                        bodyA=self.fuselage,
                        bodyB=thruster,
                        localAnchorA=(
                            (side * (BOOSTER_RADIUS + w / 3)) / SI_UNITS_SCALE,
                            (BOOSTER_HEIGHT * 0.9) / SI_UNITS_SCALE),
                        localAnchorB=(0, 0),
                        referenceAngle=0,
                    )
                )
                self.sideThrusters.append(thruster)

            # ------------------------------------- NOZZLE -------------------------------------------
        NOZZLE_HEIGHT = 2
        NOZZLE_POLY = [
            (-BOOSTER_RADIUS * 0.8, NOZZLE_HEIGHT / 2), (BOOSTER_RADIUS * 0.8, NOZZLE_HEIGHT / 2),
            (NOZZLE_RADIUS, -NOZZLE_HEIGHT / 2), (-NOZZLE_RADIUS, -NOZZLE_HEIGHT / 2)
        ]
        NOZZLE_POLY = [(x / SI_UNITS_SCALE, y / SI_UNITS_SCALE) for x, y in NOZZLE_POLY]

        self.nozzle: Box2D.b2Body = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0,
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(vertices=NOZZLE_POLY),
                density=FUSELAGE_DENSITY,
                friction=0.0,
                categoryBits=0x0040,
                maskBits=0x003,  # collide only with ground
                restitution=0.0  # 0 (rigid) 0.99 (bouncy)
            ),
        )

        self.world.CreateWeldJoint(
            bodyA=self.fuselage,
            bodyB=self.nozzle,
            localAnchorA=(0, 0),
            localAnchorB=(0, (NOZZLE_HEIGHT / 2) / SI_UNITS_SCALE),  # (0, (NOZZLE_HEIGHT / 2) / SI_UNITS_SCALE),
            referenceAngle=0
        )

        return [self.fuselage, self.nozzle] + self.legs + self.sideThrusters

    def _createParticle(self, mass, x, y, ttl, radius=3.0):
        """
            Particles represents the engine thrusts
        :return:
        """
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=b2FixtureDef(
                shape=b2CircleShape(radius=radius / SI_UNITS_SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0
            )
        )
        p.ttl = ttl  # ttl is decreased with every time step to determine if the particle should be destroyed
        self.particles.append(p)
        # Check if some particles need cleaning
        self._destroyParticles(False)
        return p

    def _updateParticles(self):
        """Update particle effects to simulate exhaust or smoke."""
        for particle in self.particles:
            particle.ttl -= 0.15
        self.particles = [p for p in self.particles if p.ttl > 0.0]

    # ------------------------------------- DESTROY ------------------------------------- #
    def _destroyParticles(self, destroyAll=False):
        while self.particles and (destroyAll or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def destroy(self):
        self._destroyParticles()
        self.world.DestroyBody(self.terrain)
        self.world.DestroyBody(self.launch_pad)
        self.terrain = None
        self.particles = None
        self.launch_pad = None
        for obj in self.booster:
            self.world.DestroyBody(obj)
        self.world.contactListener = None
        self.world = None
        self.booster = None