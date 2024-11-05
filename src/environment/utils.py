from typing import Dict, List, Tuple
import logging
import numpy as np
from .constants import LAUNCH_PAD_CENTER, LAUNCH_PAD_HEIGHT, GROUND_HEIGHT
from Box2D import b2Body
import os
import yaml

def noisy(mean: float, sigma: float) -> float:
    """
    Generates a random number based on a normal distribution with given mean and standard deviation.

    :param mean: Mean of the distribution.
    :param sigma: Standard deviation of the distribution.
    :return: A random float from the specified normal distribution.
    """
    return np.random.normal(mean, sigma)


def points_distance(A: Tuple[float, float], B: Tuple[float, float]) -> Tuple[Tuple[float, float], float]:
    """
    Calculates the relative position and Euclidean distance from point A to point B.

    :param A: Tuple of X and Y coordinates for point A.
    :param B: Tuple of X and Y coordinates for point B (target).
    :return: A tuple containing:
             - Relative coordinates (X, Y) from point A to point B.
             - Absolute Euclidean distance from point A to point B.
    """
    rel_pos = (A[0] - B[0], A[1] - B[1])  # Calculate relative position
    distance = np.linalg.norm(rel_pos)     # Calculate Euclidean distance
    return rel_pos, distance


def calculateCG(bodies: List[b2Body]) -> Tuple[float, float]:
    """
    Calculates the composite center of gravity (CG) for a list of bodies based on their mass and world center.

    :param bodies: List of Box2D bodies with mass and world center attributes.
    :return: The (X, Y) coordinates of the composite center of gravity.
    """
    composite_mass = sum(body.mass for body in bodies)
    if composite_mass == 0:
        logging.warning("Composite mass is zero; returning origin as center of gravity.")
        return (0.0, 0.0)

    weighted_positions = [
        (body.mass * body.worldCenter[0], body.mass * body.worldCenter[1]) for body in bodies
    ]

    center_of_gravity = (
        sum(x for x, _ in weighted_positions) / composite_mass,
        sum(y for _, y in weighted_positions) / composite_mass
    )
    return center_of_gravity


def m_dot(F: float, Ve: float, Pe: float, _Pa: float, Ae: float, mixRatio: float) -> Tuple[float, float, float]:
    """
    Calculates the total mass flow rate, fuel mass flow rate, and oxidizer mass flow rate.

    :param F: Thrust produced by the engine (Newtons).
    :param Ve: Effective exhaust velocity at nozzle exit (m/s).
    :param Pe: Exhaust pressure at nozzle exit (Pa).
    :param _Pa: Ambient pressure (Pa).
    :param Ae: Exit area of the nozzle (m^2).
    :param mixRatio: Mass flow rate ratio of oxidizer to fuel.
    :return: A tuple containing:
             - Total mass flow rate (kg/s).
             - Fuel mass flow rate (kg/s).
             - Oxidizer mass flow rate (kg/s).
    """
    _m_dot = (F - ((Pe - _Pa) * Ae)) / Ve
    fuel_flow = _m_dot / (1 + mixRatio)
    oxidizer_flow = _m_dot * (mixRatio / (1 + mixRatio))
    return _m_dot, fuel_flow, oxidizer_flow
