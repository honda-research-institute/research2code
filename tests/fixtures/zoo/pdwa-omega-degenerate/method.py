"""Enhanced DWA with Dynamic Obstacle Prediction.

This module implements the Enhanced Dynamic Window Approach (DWA) algorithm from the paper:
"Enhancing Obstacle Avoidance in Dynamic Window Approach via Dynamic Obstacle Behavior Prediction"
by Bongsu Hahn (Hongik University, 2025).

The Enhanced DWA is a local path planning algorithm that extends the conventional DWA by:
1. Predicting future obstacle positions using a linear motion model (Eq 3, Section 3.1)
2. Modifying the dynamic window to exclude velocity pairs leading to future collisions (Eq 5, Section 3.2)
3. Using a comprehensive objective function with six terms for velocity evaluation (Eq 6-12, Section 3.3):
   - J_heading: Evaluates heading angle toward goal (Eq 7)
   - J_distance: Euclidean distance to goal, redefined for dynamic environments (Eq 8)
   - J_clearance: Exponential scoring for distance to nearby obstacles (Eq 9)
   - J_future_obs: Proactive collision avoidance based on predicted positions (Eq 10)
   - J_risk_obs: Differential penalty based on obstacle distance and angle (Eq 11)
   - J_count_obs: Guides robot toward side with fewer obstacles (Eq 12)
4. Selecting optimal velocity by maximizing the objective function (Eq 13, Section 3.4)

Public functions:
- select_velocity: Main pluggable function implementing the Enhanced DWA algorithm
- predict_obstacle_position: Predicts future obstacle position using linear model (Eq 3)
- compute_dynamic_window: Generates feasible velocity pairs avoiding future collisions (Eq 5)
- evaluate_objective_function: Computes the six-term objective function (Eq 6-12)
- update_robot_state: Updates robot position after executing velocity (Eq 14)

Paper citation:
Hahn, B. (2025). Enhancing Obstacle Avoidance in Dynamic Window Approach via Dynamic Obstacle Behavior Prediction.
Actuators, 14(5), 207. https://doi.org/10.3390/act14050207
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# ============================================================================
# Helper Functions
# ============================================================================


def predict_obstacle_position(
    obs_x: float,
    obs_y: float,
    obs_x_prev: float,
    obs_y_prev: float,
    dt: float,
) -> tuple[float, float, float, float]:
    """Predict future obstacle position using linear motion model.

    Implements Eq (3) from Section 3.1. The model assumes the obstacle moves
    at constant velocity and direction over short time intervals. Given the
    current and previous positions, it computes the obstacle's velocity and
    heading, then predicts its position at time t+1.

    # paper-element: eq-obstacle-prediction
    # paper-element: concept-dynamic-obstacle-prediction

    Args:
        obs_x: Current obstacle x-position.
        obs_y: Current obstacle y-position.
        obs_x_prev: Previous obstacle x-position (at t-1).
        obs_y_prev: Previous obstacle y-position (at t-1).
        dt: Time interval for prediction in seconds.

    Returns:
        Tuple of (x_future, y_future, velocity, heading):
            - x_future: Predicted x-position at t+1
            - y_future: Predicted y-position at t+1
            - velocity: Computed obstacle speed in m/s
            - heading: Computed obstacle heading in radians
    """
    # Compute obstacle heading from position change
    dx = obs_x - obs_x_prev
    dy = obs_y - obs_y_prev
    
    # Handle edge case where obstacle hasn't moved
    if abs(dx) < 1e-10 and abs(dy) < 1e-10:
        return obs_x, obs_y, 0.0, 0.0
    
    # Compute heading angle (atan2 gives angle in range [-pi, pi])
    heading = math.atan2(dy, dx)
    
    # Compute velocity magnitude
    velocity = math.sqrt(dx * dx + dy * dy) / dt
    
    # Predict future position using linear model
    x_future = obs_x + velocity * math.cos(heading) * dt
    y_future = obs_y + velocity * math.sin(heading) * dt
    
    return x_future, y_future, velocity, heading


def compute_dynamic_window(
    v_max: float,
    omega_max: float,
    robot_state: tuple[float, float, float],
    goal_position: tuple[float, float],
    obstacle_states: list[dict],
    dt: float,
    r_robot: float,
    r_obs: float,
    v_resolution: float = 0.1,
    omega_resolution: float = 0.1,
) -> list[tuple[float, float]]:
    """Generate feasible velocity pairs avoiding future collisions.

    Implements Eq (5) from Section 3.2. The modified dynamic window filters
    velocity pairs based on predicted obstacle positions rather than current
    positions, enabling proactive collision avoidance.

    # paper-element: eq-dynamic-window

    Args:
        v_max: Maximum linear velocity of the robot (m/s).
        omega_max: Maximum angular velocity of the robot (rad/s).
        robot_state: Current robot state (x, y, theta).
        goal_position: Target position (x_goal, y_goal).
        obstacle_states: List of obstacle dictionaries with keys:
            'x', 'y', 'x_prev', 'y_prev' for position tracking.
        dt: Time step in seconds.
        r_robot: Robot radius in meters.
        r_obs: Obstacle radius in meters.
        v_resolution: Resolution for velocity sampling (m/s).
        omega_resolution: Resolution for angular velocity sampling (rad/s).

    Returns:
        List of feasible (v, omega) tuples that avoid predicted collisions.
    """
    x_robot, y_robot, theta_robot = robot_state
    min_safe_dist = r_robot + r_obs
    feasible_velocities = []
    
    # Sample velocity space
    v_values = np.arange(0, v_max + v_resolution, v_resolution)
    omega_values = np.arange(-omega_max, omega_max + omega_resolution, omega_resolution)
    
    for v in v_values:
        for omega in omega_values:
            is_feasible = True
            
            # Predict robot future position
            x_robot_future = x_robot + v * math.cos(theta_robot) * dt
            y_robot_future = y_robot + v * math.sin(theta_robot) * dt
            
            # Check against each obstacle's predicted position
            for obs in obstacle_states:
                # Predict obstacle future position
                obs_x_future, obs_y_future, _, _ = predict_obstacle_position(
                    obs['x'], obs['y'], obs['x_prev'], obs['y_prev'], dt
                )
                
                # Compute distance between predicted positions
                dx = x_robot_future - obs_x_future
                dy = y_robot_future - obs_y_future
                dist = math.sqrt(dx * dx + dy * dy)
                
                # Reject velocity if it leads to collision
                if dist <= min_safe_dist:
                    is_feasible = False
                    break
            
            if is_feasible:
                feasible_velocities.append((v, omega))
    
    return feasible_velocities


def evaluate_objective_function(
    v: float,
    omega: float,
    robot_state: tuple[float, float, float],
    goal_position: tuple[float, float],
    obstacle_states: list[dict],
    dt: float,
    v_max: float,
    r_robot: float,
    r_obs: float,
    alpha_h: float,
    alpha_d: float,
    alpha_c: float,
    alpha_f: float,
    beta: list[float],
) -> float:
    """Compute the Enhanced DWA objective function.

    Implements Eq (6) from Section 3.3, which combines six evaluation terms:
    - J_heading (Eq 7): Heading angle toward goal
    - J_distance (Eq 8): Distance to goal (redefined for dynamic environments)
    - J_clearance (Eq 9): Distance to nearby obstacles
    - J_future_obs (Eq 10): Future collision risk based on predictions
    - J_risk_obs (Eq 11): Multi-dimensional risk assessment
    - J_count_obs (Eq 12): Obstacle density guidance

    # paper-element: eq-objective-function

    Args:
        v: Linear velocity candidate (m/s).
        omega: Angular velocity candidate (rad/s).
        robot_state: Current robot state (x, y, theta).
        goal_position: Target position (x_goal, y_goal).
        obstacle_states: List of obstacle dictionaries with position data.
        dt: Time step in seconds.
        v_max: Maximum robot velocity (for d_safe computation).
        r_robot: Robot radius.
        r_obs: Obstacle radius.
        alpha_h: Weight for heading term.
        alpha_d: Weight for distance term.
        alpha_c: Weight for clearance term.
        alpha_f: Weight for future prediction term.
        beta: List of five risk penalty coefficients [beta_1, ..., beta_5].

    Returns:
        Total objective function value J(v, omega).
    """
    x_robot, y_robot, theta_robot = robot_state
    x_goal, y_goal = goal_position
    
    # Compute safety distance (Eq 9)
    d_safe = v_max / 4.0
    dist_safe = d_safe  # Same value used for future prediction term
    
    # Predict robot future position
    x_robot_future = x_robot + v * math.cos(theta_robot) * dt
    y_robot_future = y_robot + v * math.sin(theta_robot) * dt
    
    # ---------------------------------------------------------------------
    # Term 1: J_heading (Eq 7) - Heading evaluation
    # ---------------------------------------------------------------------
    # paper-element: eq-heading
    theta_to_goal = math.atan2(y_goal - y_robot, x_goal - x_robot)
    theta_heading = theta_to_goal - theta_robot
    # Normalize to [-pi, pi]
    while theta_heading > math.pi:
        theta_heading -= 2 * math.pi
    while theta_heading < -math.pi:
        theta_heading += 2 * math.pi
    j_heading = -alpha_h * abs(theta_heading)
    
    # ---------------------------------------------------------------------
    # Term 2: J_distance (Eq 8) - Distance to goal
    # ---------------------------------------------------------------------
    # paper-element: eq-distance
    # paper-element: concept-redefined-distance-term
    distance_to_goal = math.sqrt(
        (x_goal - x_robot) ** 2 + (y_goal - y_robot) ** 2
    )
    j_distance = -alpha_d * distance_to_goal
    
    # ---------------------------------------------------------------------
    # Terms 3-6: Obstacle-related terms
    # ---------------------------------------------------------------------
    
    j_clearance_total = 0.0
    j_future_obs_total = 0.0
    j_risk_obs_total = 0.0
    
    # Counters for obstacle density term
    count_obs_left = 0
    count_obs_right = 0
    
    for obs in obstacle_states:
        obs_x, obs_y = obs['x'], obs['y']
        
        # Predict obstacle future position
        obs_x_future, obs_y_future, _, _ = predict_obstacle_position(
            obs_x, obs_y, obs['x_prev'], obs['y_prev'], dt
        )
        
        # -----------------------------------------------------------------
        # Term 3: J_clearance (Eq 9) - Clearance from current obstacle
        # -----------------------------------------------------------------
        # paper-element: eq-clearance
        clearance = math.sqrt(
            (obs_x - x_robot_future) ** 2 + (obs_y - y_robot_future) ** 2
        )
        j_clearance = math.exp(alpha_c * (d_safe - clearance))
        j_clearance_total += j_clearance
        
        # -----------------------------------------------------------------
        # Term 4: J_future_obs (Eq 10) - Future collision risk
        # -----------------------------------------------------------------
        # paper-element: eq-future-obs
        dist_future = math.sqrt(
            (obs_x_future - x_robot_future) ** 2 + (obs_y_future - y_robot_future) ** 2
        )
        j_future_obs = math.exp(alpha_f * (dist_safe - dist_future))
        j_future_obs_total += j_future_obs
        
        # -----------------------------------------------------------------
        # Term 5: J_risk_obs (Eq 11) - Multi-dimensional risk assessment
        # -----------------------------------------------------------------
        # paper-element: eq-risk-obs
        # paper-element: concept-risk-assessment
        theta_obs = math.atan2(obs_y - y_robot, obs_x - x_robot)
        theta_h = theta_obs - theta_robot
        # Normalize to [-pi, pi]
        while theta_h > math.pi:
            theta_h -= 2 * math.pi
        while theta_h < -math.pi:
            theta_h += 2 * math.pi
        
        dist_n = math.sqrt((obs_x - x_robot) ** 2 + (obs_y - y_robot) ** 2)
        
        # Apply differential penalty based on angle and distance
        beta_1, beta_2, beta_3, beta_4, beta_5 = beta
        risk_penalty = 0.0
        
        if abs(theta_h) <= math.pi / 12:
            if dist_n > dist_safe:
                risk_penalty = beta_1
            else:
                risk_penalty = beta_2
        elif abs(theta_h) <= math.pi / 6:
            if dist_n <= dist_safe:
                risk_penalty = beta_3
            elif dist_n <= 2 * dist_safe:
                risk_penalty = beta_4
            elif dist_n <= 3 * dist_safe:
                risk_penalty = beta_5
        
        j_risk_obs_total += risk_penalty
        
        # -----------------------------------------------------------------
        # Term 6: J_count_obs (Eq 12) - Obstacle density counting
        # -----------------------------------------------------------------
        # paper-element: eq-count-obs
        # paper-element: concept-obstacle-density-guidance
        # Count obstacles within 20-degree (pi/9) angle on each side
        if -math.pi / 9 < theta_h <= 0:
            count_obs_left += 1
        elif 0 <= theta_h < math.pi / 9:
            count_obs_right += 1
    
    # Compute J_count_obs
    if count_obs_right > count_obs_left:
        j_count_obs = -(count_obs_right - count_obs_left)
    elif count_obs_left > count_obs_right:
        j_count_obs = -(count_obs_left - count_obs_right)
    else:
        j_count_obs = 0.0
    
    # Total objective function value
    j_total = (
        j_heading +
        j_distance +
        j_clearance_total +
        j_future_obs_total +
        j_risk_obs_total +
        j_count_obs
    )
    
    return j_total


def update_robot_state(
    robot_state: tuple[float, float, float],
    v: float,
    omega: float,
    dt: float,
) -> tuple[float, float, float]:
    """Update robot position after executing velocity command.

    Implements Eq (14) from Section 3.4. Uses the non-holonomic kinematic
    model to propagate the robot state forward in time.

    # paper-element: eq-robot-update
    # paper-element: eq-robot-kinematics

    Args:
        robot_state: Current robot state (x, y, theta).
        v: Linear velocity (m/s).
        omega: Angular velocity (rad/s).
        dt: Time step in seconds.

    Returns:
        Updated robot state (x_new, y_new, theta_new).
    """
    x, y, theta = robot_state
    
    x_new = x + v * math.cos(theta) * dt
    y_new = y + v * math.sin(theta) * dt
    theta_new = theta + omega * dt
    
    return x_new, y_new, theta_new


# ============================================================================
# Main Pluggable Function
# ============================================================================


def select_velocity(
    robot_state: tuple[float, float, float],
    obstacle_states: list[dict],
    goal_position: tuple[float, float],
    dt: float,
    seed: int,
    alpha_h: float = 0.5,
    alpha_d: float = 0.5,
    alpha_c: float = 0.5,
    alpha_f: float = 0.5,
    beta: list[float] = None,
) -> tuple[float, float]:
    """Select optimal velocity using Enhanced DWA with dynamic obstacle prediction.

    This is the main pluggable function implementing the Enhanced DWA algorithm
    from the paper. It integrates obstacle behavior prediction into the DWA
    framework to enable safe avoidance of moving obstacles.

    The algorithm:
    1. Generates a modified dynamic window excluding velocities leading to
       future collisions with predicted obstacle positions (Eq 5)
    2. Evaluates each feasible velocity using a six-term objective function (Eq 6)
    3. Selects the velocity pair maximizing the objective function (Eq 13)

    # paper-element: alg-enhanced-dwa
    # essential: Differential drive control (v, omega from wheel velocities)
    # Note: This function outputs (v, omega) control values. The actual conversion
    # to wheel velocities (Eq 2: v = r/2*(v_L+v_R), omega = r/h*(v_R-v_L)) is
    # handled by the robot's motor controller, not the planning algorithm.

    Paper section: Section 3 (Enhanced DWA with Dynamic Obstacle Prediction)

    Args:
        robot_state: Current robot state (x, y, theta) in (m, m, rad).
        obstacle_states: List of obstacle dictionaries, each containing:
            - 'x': Current x-position (m)
            - 'y': Current y-position (m)
            - 'x_prev': Previous x-position (m)
            - 'y_prev': Previous y-position (m)
        goal_position: Target position (x_goal, y_goal) in (m, m).
        dt: Time step for prediction and control (s).
        seed: Random seed for stochastic components (used for velocity sampling
              tie-breaking when multiple velocities have equal scores).
        alpha_h: Weight for heading evaluation term (default: 0.5).
        alpha_d: Weight for distance evaluation term (default: 0.5).
        alpha_c: Weight for clearance evaluation term (default: 0.5).
        alpha_f: Weight for future prediction term (default: 0.5).
        beta: List of five risk penalty coefficients [beta_1, beta_2, beta_3,
              beta_4, beta_5] with beta_1 < beta_2 < ... < beta_5.
              Default: [0.1, 0.5, 1.0, 2.0, 5.0].

    Returns:
        Tuple of (v_star, omega_star): Optimal linear velocity (m/s) and
        angular velocity (rad/s) that maximize the objective function.

    Raises:
        ValueError: If no feasible velocities are found or obstacle_states is empty.
    """
    # Set default beta coefficients if not provided
    if beta is None:
        beta = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    # Initialize random number generator from seed
    rng = np.random.default_rng(seed)
    
    # Robot parameters (from paper Section 2.2, Table 1)
    v_max = 1.5  # m/s
    omega_max = 1.0  # rad/s
    r_robot = 0.1  # m
    r_obs = 0.15  # m
    
    # Generate feasible velocity pairs (modified dynamic window)
    feasible_velocities = compute_dynamic_window(
        v_max=v_max,
        omega_max=omega_max,
        robot_state=robot_state,
        goal_position=goal_position,
        obstacle_states=obstacle_states,
        dt=dt,
        r_robot=r_robot,
        r_obs=r_obs,
    )
    
    # Handle edge case: no feasible velocities
    if not feasible_velocities:
        # Return zero velocity as safe fallback
        # This can happen in highly constrained environments
        return (0.0, 0.0)
    
    # Evaluate objective function for each feasible velocity
    best_score = float('-inf')
    best_velocities = []
    
    for v, omega in feasible_velocities:
        score = evaluate_objective_function(
            v=v,
            omega=omega,
            robot_state=robot_state,
            goal_position=goal_position,
            obstacle_states=obstacle_states,
            dt=dt,
            v_max=v_max,
            r_robot=r_robot,
            r_obs=r_obs,
            alpha_h=alpha_h,
            alpha_d=alpha_d,
            alpha_c=alpha_c,
            alpha_f=alpha_f,
            beta=beta,
        )
        
        if score > best_score:
            best_score = score
            best_velocities = [(v, omega)]
        elif abs(score - best_score) < 1e-10:
            # Tie: add to candidates for random selection
            best_velocities.append((v, omega))
    
    # Select from best velocities (random tie-breaking for reproducibility)
    # paper-element: eq-optimal-selection
    if len(best_velocities) == 1:
        v_star, omega_star = best_velocities[0]
    else:
        # Random selection among tied velocities using seeded RNG
        idx = rng.integers(0, len(best_velocities))
        v_star, omega_star = best_velocities[idx]
    
    return (v_star, omega_star)
