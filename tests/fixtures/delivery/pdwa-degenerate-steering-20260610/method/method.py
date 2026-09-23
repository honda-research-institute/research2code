"""
PDWA (Predictive Dynamic Window Approach) - Enhanced DWA with Dynamic Obstacle Behavior Prediction.

This module implements the local path planning algorithm from:
"Enhancing Obstacle Avoidance in Dynamic Window Approach via Dynamic Obstacle Behavior Prediction"
by Bongsu Hahn, Actuators 2025.

The method enhances conventional DWA by:
1. Predicting future obstacle positions using a linear model (Eq. 3)
2. Constraining the dynamic window based on predicted obstacle positions (Eq. 5)
3. Using a six-term risk-aware evaluation function (Eq. 6-12) to select optimal velocity

Public functions:
- plan: The main pluggable planner function
- predict_obstacle_position: Linear obstacle position prediction (Eq. 3)
- compute_j_heading: Heading alignment evaluation (Eq. 7)
- compute_j_distance: Goal distance evaluation (Eq. 8)
- compute_j_clearance: Obstacle clearance evaluation (Eq. 9)
- compute_j_future_obs: Future obstacle position evaluation (Eq. 10)
- compute_j_risk_obs: Risk-based obstacle penalty (Eq. 11)
- compute_j_count_obs: Obstacle count-based directional penalty (Eq. 12)

# paper-element: alg-enhanced-dwa
# paper-element: concept-prediction-model
# paper-element: concept-risk-assessment
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, atan2, cos, sin, sqrt, stack, exp, abs as torch_abs


@dataclass
class PlanResult:
    """Result of the planning process.
    
    Attributes:
        trajectory: List of (state, control) tuples representing the planned path.
                   Each state is (x, y, theta), each control is (v, omega).
        status: One of "success", "timeout", "infeasible", "error".
        stats: Dictionary with planning statistics (iterations, wall_time, etc.).
    """
    trajectory: list[tuple[Tensor, Tensor]] | None
    status: str
    stats: dict[str, Any]


# paper-element: eq-obstacle-prediction
def predict_obstacle_position(
    obs_current: Tensor,
    obs_previous: Tensor,
    dt: float
) -> tuple[Tensor, float, float]:
    """Predict obstacle future position using linear prediction model (Eq. 3).
    
    Infers obstacle velocity and heading from two consecutive position observations,
    then predicts the next position assuming constant velocity and heading.
    
    Args:
        obs_current: Current obstacle position (x, y) with shape (2,)
        obs_previous: Previous obstacle position (x, y) with shape (2,)
        dt: Time step in seconds
        
    Returns:
        Tuple of (predicted_position, velocity, heading) where:
        - predicted_position: Next obstacle position (x, y) with shape (2,)
        - velocity: Obstacle speed in m/s
        - heading: Obstacle direction in radians
        
    # paper-element: eq-obstacle-prediction
    # paper-element: concept-prediction-model
    """
    # Compute displacement
    dx = obs_current[0] - obs_previous[0]
    dy = obs_current[1] - obs_previous[1]
    
    # Compute heading using atan2 (Eq. 3)
    theta_obs = torch.atan2(dy, dx)
    
    # Compute velocity as displacement magnitude over dt (Eq. 3)
    v_obs = torch.sqrt(dx**2 + dy**2) / dt
    
    # Predict next position (Eq. 3)
    x_next = obs_current[0] + v_obs * torch.cos(theta_obs) * dt
    y_next = obs_current[1] + v_obs * torch.sin(theta_obs) * dt
    
    predicted_position = torch.stack([x_next, y_next])
    
    return predicted_position, v_obs.item(), theta_obs.item()


# paper-element: eq-j-heading
def compute_j_heading(
    robot_state: Tensor,
    goal: Tensor,
    alpha_h: float = 0.5
) -> float:
    """Compute heading alignment evaluation term (Eq. 7).
    
    Evaluates how aligned the robot's current heading is with the direction to the goal.
    A negative score proportional to the absolute heading error encourages alignment.
    
    Args:
        robot_state: Robot state (x, y, theta) with shape (3,)
        goal: Goal position (x, y) with shape (2,)
        alpha_h: Weight factor for heading term
        
    Returns:
        Heading evaluation score (negative, higher is better)
        
    # paper-element: eq-j-heading
    """
    x_robot, y_robot, theta_robot = robot_state[0], robot_state[1], robot_state[2]
    x_goal, y_goal = goal[0], goal[1]
    
    # Compute heading angle toward goal (Eq. 7)
    theta_heading = torch.atan2(y_goal - y_robot, x_goal - x_robot) - theta_robot
    
    # Normalize to [-pi, pi]
    theta_heading = torch.atan2(torch.sin(theta_heading), torch.cos(theta_heading))
    
    # Compute J_heading (Eq. 7)
    j_heading = -alpha_h * torch.abs(theta_heading)
    
    return j_heading.item()


# paper-element: eq-j-distance
def compute_j_distance(
    robot_state: Tensor,
    goal: Tensor,
    alpha_d: float = 0.5
) -> float:
    """Compute goal distance evaluation term (Eq. 8).
    
    Evaluates the Euclidean distance between the robot's current position and the goal.
    Redefined in this work as distance (not speed) to incentivize reducing distance.
    
    Args:
        robot_state: Robot state (x, y, theta) with shape (3,)
        goal: Goal position (x, y) with shape (2,)
        alpha_d: Weight factor for distance term
        
    Returns:
        Distance evaluation score (negative, higher is better)
        
    # paper-element: eq-j-distance
    """
    x_robot, y_robot = robot_state[0], robot_state[1]
    x_goal, y_goal = goal[0], goal[1]
    
    # Compute Euclidean distance to goal (Eq. 8)
    dist = torch.sqrt((x_goal - x_robot)**2 + (y_goal - y_robot)**2)
    
    # Compute J_distance (Eq. 8)
    j_distance = -alpha_d * dist
    
    return j_distance.item()


# paper-element: eq-j-clearance
def compute_j_clearance(
    robot_state: Tensor,
    control: Tensor,
    dt: float,
    obstacle_positions: list[Tensor],
    d_safe: float,
    alpha_c: float = 0.5,
    dynamics=None
) -> float:
    """Compute obstacle clearance evaluation term (Eq. 9).
    
    Exponential penalty based on distance between robot's predicted next position
    and each obstacle's current position.
    
    Args:
        robot_state: Robot state (x, y, theta) with shape (3,)
        control: Control input (v, omega) with shape (2,)
        dt: Time step in seconds
        obstacle_positions: List of obstacle positions, each (x, y) with shape (2,)
        d_safe: Safety distance for collision avoidance (V_max / 4)
        alpha_c: Weight factor for clearance term
        dynamics: Dynamics object with step() method
        
    Returns:
        Clearance evaluation score (exponential penalty, lower is better)
        
    # paper-element: eq-j-clearance
    """
    # Predict robot's next position
    robot_next = dynamics.step(robot_state, control, dt)
    x_robot_next, y_robot_next = robot_next[0], robot_next[1]
    
    j_clearance = 0.0
    for obs_pos in obstacle_positions:
        x_obs, y_obs = obs_pos[0], obs_pos[1]
        
        # Compute clearance distance (Eq. 9)
        clearance = torch.sqrt((x_obs - x_robot_next)**2 + (y_obs - y_robot_next)**2)
        
        # Compute exponential penalty (Eq. 9)
        j_clearance += torch.exp(alpha_c * (d_safe - clearance)).item()
    
    return j_clearance


# paper-element: eq-j-future-obs
def compute_j_future_obs(
    robot_state: Tensor,
    control: Tensor,
    dt: float,
    obstacle_current: list[Tensor],
    obstacle_previous: list[Tensor],
    d_safe: float,
    alpha_f: float = 0.5,
    dynamics=None
) -> float:
    """Compute future obstacle position evaluation term (Eq. 10).
    
    Exponential penalty based on predicted distance between robot's future position
    and each obstacle's future position. Enables proactive collision avoidance.
    
    Args:
        robot_state: Robot state (x, y, theta) with shape (3,)
        control: Control input (v, omega) with shape (2,)
        dt: Time step in seconds
        obstacle_current: List of current obstacle positions, each (x, y) with shape (2,)
        obstacle_previous: List of previous obstacle positions, each (x, y) with shape (2,)
        d_safe: Safety distance for collision avoidance
        alpha_f: Weight factor for future obstacle term
        dynamics: Dynamics object with step() method
        
    Returns:
        Future obstacle evaluation score (exponential penalty, lower is better)
        
    # paper-element: eq-j-future-obs
    """
    # Predict robot's next position
    robot_next = dynamics.step(robot_state, control, dt)
    x_robot_next, y_robot_next = robot_next[0], robot_next[1]
    
    j_future_obs = 0.0
    for obs_curr, obs_prev in zip(obstacle_current, obstacle_previous):
        # Predict obstacle's next position
        obs_next, _, _ = predict_obstacle_position(obs_curr, obs_prev, dt)
        x_obs_next, y_obs_next = obs_next[0], obs_next[1]
        
        # Compute future distance (Eq. 10)
        dist_future = torch.sqrt((x_obs_next - x_robot_next)**2 + (y_obs_next - y_robot_next)**2)
        
        # Compute exponential penalty (Eq. 10)
        j_future_obs += torch.exp(alpha_f * (d_safe - dist_future)).item()
    
    return j_future_obs


# paper-element: eq-j-risk-obs
def compute_j_risk_obs(
    robot_state: Tensor,
    obstacle_positions: list[Tensor],
    d_safe: float,
    beta_1: float = 1.0,
    beta_2: float = 2.0,
    beta_3: float = 3.0,
    beta_4: float = 4.0,
    beta_5: float = 5.0
) -> float:
    """Compute risk-based obstacle penalty term (Eq. 11).
    
    Piecewise penalty function that assigns different risk levels based on each
    obstacle's relative angle and distance to the robot. Five tiers of penalty
    (beta_1 < beta_2 < beta_3 < beta_4 < beta_5) apply progressively.
    
    Args:
        robot_state: Robot state (x, y, theta) with shape (3,)
        obstacle_positions: List of obstacle positions, each (x, y) with shape (2,)
        d_safe: Safety distance for collision avoidance
        beta_1 through beta_5: Risk penalty levels (monotonically increasing)
        
    Returns:
        Risk evaluation score (sum of penalties, lower is better)
        
    # paper-element: eq-j-risk-obs
    # paper-element: concept-risk-assessment
    """
    x_robot, y_robot, theta_robot = robot_state[0], robot_state[1], robot_state[2]
    
    j_risk = 0.0
    
    # Angle thresholds (Eq. 11)
    pi_12 = np.pi / 12  # 15 degrees
    pi_6 = np.pi / 6    # 30 degrees
    
    for obs_pos in obstacle_positions:
        x_obs, y_obs = obs_pos[0], obs_pos[1]
        
        # Compute relative angle (Eq. 11)
        theta_H = torch.atan2(y_obs - y_robot, x_obs - x_robot) - theta_robot
        # Normalize to [-pi, pi]
        theta_H = torch.atan2(torch.sin(theta_H), torch.cos(theta_H))
        theta_H_abs = torch.abs(theta_H).item()
        
        # Compute distance (Eq. 11)
        d = torch.sqrt((x_obs - x_robot)**2 + (y_obs - y_robot)**2).item()
        
        # Apply piecewise penalty function (Eq. 11)
        if theta_H_abs <= pi_12 and d > d_safe:
            penalty = beta_1
        elif theta_H_abs <= pi_12 and d <= d_safe:
            penalty = beta_2
        elif theta_H_abs <= pi_6 and d <= d_safe:
            penalty = beta_3
        elif theta_H_abs <= pi_6 and d <= 2 * d_safe:
            penalty = beta_4
        elif theta_H_abs <= pi_6 and d <= 3 * d_safe:
            penalty = beta_5
        else:
            penalty = 0.0
        
        j_risk += penalty
    
    return j_risk


# paper-element: eq-j-count-obs
def compute_j_count_obs(
    robot_state: Tensor,
    obstacle_positions: list[Tensor]
) -> float:
    """Compute obstacle count-based directional penalty term (Eq. 12).
    
    Evaluates the number of obstacles on the left and right sides of the robot's
    movement direction within a 20-degree forward field of view. Encourages
    the robot to steer toward the side with fewer obstacles.
    
    Args:
        robot_state: Robot state (x, y, theta) with shape (3,)
        obstacle_positions: List of obstacle positions, each (x, y) with shape (2,)
        
    Returns:
        Directional penalty score (negative if imbalance, 0 if balanced)
        
    # paper-element: eq-j-count-obs
    """
    x_robot, y_robot, theta_robot = robot_state[0], robot_state[1], robot_state[2]
    
    # Angle threshold: pi/9 = 20 degrees (Eq. 12)
    pi_9 = np.pi / 9
    
    count_L = 0  # Obstacles on left side
    count_R = 0  # Obstacles on right side
    
    for obs_pos in obstacle_positions:
        x_obs, y_obs = obs_pos[0], obs_pos[1]
        
        # Compute relative angle
        theta_H = torch.atan2(y_obs - y_robot, x_obs - x_robot) - theta_robot
        # Normalize to [-pi, pi]
        theta_H = torch.atan2(torch.sin(theta_H), torch.cos(theta_H))
        theta_H_val = theta_H.item()
        
        # Count obstacles in left sector (Eq. 12): -pi/9 < theta_H <= 0
        if -pi_9 < theta_H_val <= 0:
            count_L += 1
        
        # Count obstacles in right sector (Eq. 12): 0 <= theta_H < pi/9
        if 0 <= theta_H_val < pi_9:
            count_R += 1
    
    # Compute penalty (Eq. 12)
    if count_R > count_L:
        j_count = -(count_R - count_L)
    elif count_L > count_R:
        j_count = -(count_L - count_R)
    else:
        j_count = 0.0
    
    return j_count


def plan(
    start,
    goal,
    environment,
    dynamics,
    seed,
    dt: float = 0.1,
    d_safe: float = None,
    beta_1: float = 1.0,
    beta_2: float = 2.0,
    beta_3: float = 3.0,
    beta_4: float = 4.0,
    beta_5: float = 5.0
) -> PlanResult:
    """PDWA planner with dynamic obstacle behavior prediction.
    
    Implements the complete PDWA algorithm:
    1. Predicts future obstacle positions (Eq. 3)
    2. Constrains dynamic window based on predicted positions (Eq. 5)
    3. Evaluates candidate velocities using six-term objective (Eq. 6-12)
    4. Selects argmax and updates robot state (Eq. 13-14)
    
    Args:
        start: Initial robot state (x, y, theta) with shape (3,)
        goal: Goal position (x, y) with shape (2,)
        environment: Environment object from data.py with obstacles field:
            - obstacles: List[Dict[str, Any]] where each dict has keys like
              'type', 'x', 'y', 'radius' (circles) or 'xmin', 'xmax', 'ymin', 'ymax' (rectangles)
        dynamics: Dynamics object with step(state, control, dt) method
        seed: Random seed for stochastic operations
        dt: Time step in seconds
        d_safe: Safety distance (default: V_max / 4)
        beta_1 through beta_5: Risk penalty levels for j_risk_obs
        
    Returns:
        PlanResult with trajectory, status, and statistics
        
    # paper-element: alg-enhanced-dwa
    # paper-element: eq-optimal-path-selection
    # paper-element: eq-robot-position-update
    # essential: Unicycle kinematics with Euler integration
    # essential: Circular robot body for collision checking
    """
    # Initialize random number generator
    rng = np.random.default_rng(seed)
    
    # Convert inputs to tensors if needed
    if not isinstance(start, Tensor):
        start = torch.tensor(start, dtype=torch.float64)
    if not isinstance(goal, Tensor):
        goal = torch.tensor(goal, dtype=torch.float64)
    
    # Initialize robot state
    robot_state = start.clone()
    
    # Initialize trajectory
    trajectory = []
    
    # Compute d_safe if not provided (Eq. 9)
    if d_safe is None:
        d_safe = dynamics.V_max / 4
    
    # Get dynamics parameters
    V_max = dynamics.V_max
    omega_max = dynamics.omega_max
    r_robot = dynamics.r_robot if hasattr(dynamics, 'r_robot') else 0.1
    
    # Planning loop parameters
    max_iterations = 500  # paper-fidelity: reduced from paper scale for smoke demo
    v_grid_resolution = 0.1  # paper-fidelity: coarser grid for smoke scale
    omega_grid_resolution = 0.1  # paper-fidelity: coarser grid for smoke scale
    
    # Weight factors (tuned empirically in paper)
    alpha_h = 0.5
    alpha_d = 0.5
    alpha_c = 0.5
    alpha_f = 0.5
    
    iteration = 0
    converged = False
    
    # Track previous obstacle positions internally (Environment doesn't have get_obstacle_previous_positions)
    obstacle_previous = None
    
    while iteration < max_iterations:
        # Check if goal reached
        dist_to_goal = torch.sqrt(
            (goal[0] - robot_state[0])**2 + (goal[1] - robot_state[1])**2
        )
        if dist_to_goal < 0.3:  # Within 0.3m of goal
            converged = True
            break
        
        # Extract current obstacle positions from environment.obstacles
        # Each obstacle dict has geometry keys (type, x, y, radius for circles; xmin, xmax, ymin, ymax for rectangles)
        obstacle_current = []
        for obs in environment.obstacles:
            obs_type = obs.get("type", "circle")
            if obs_type == "circle":
                # Circle: use center (x, y)
                pos = torch.tensor([obs["x"], obs["y"]], dtype=torch.float64)
            elif obs_type == "rectangle":
                # Rectangle: use center point
                pos = torch.tensor([
                    (obs["xmin"] + obs["xmax"]) / 2.0,
                    (obs["ymin"] + obs["ymax"]) / 2.0
                ], dtype=torch.float64)
            else:
                # Default: assume circle with x, y
                pos = torch.tensor([obs.get("x", 0.0), obs.get("y", 0.0)], dtype=torch.float64)
            obstacle_current.append(pos)
        
        # Handle first timestep: if no previous positions, use current as previous
        if obstacle_previous is None:
            obstacle_previous = [obs.clone() for obs in obstacle_current]
        
        # Predict future obstacle positions (Eq. 3)
        obstacle_future = []
        for obs_curr, obs_prev in zip(obstacle_current, obstacle_previous):
            obs_next, _, _ = predict_obstacle_position(obs_curr, obs_prev, dt)
            obstacle_future.append(obs_next)
        
        # Generate velocity grid
        v_values = np.arange(0, V_max + v_grid_resolution, v_grid_resolution)
        omega_values = np.arange(-omega_max, omega_max + omega_grid_resolution, omega_grid_resolution)
        
        best_j = float('-inf')
        best_v = 0.0
        best_omega = 0.0
        feasible_count = 0
        
        # Evaluate each velocity pair
        for v in v_values:
            for omega in omega_values:
                control = torch.tensor([v, omega], dtype=torch.float64)
                
                # Predict robot's next position
                robot_next = dynamics.step(robot_state, control, dt)
                x_robot_next, y_robot_next = robot_next[0], robot_next[1]
                
                # Check collision with predicted obstacle positions (Eq. 5)
                feasible = True
                r_obs = 0.15  # Default obstacle radius
                for obs_future_pos in obstacle_future:
                    dist = torch.sqrt(
                        (obs_future_pos[0] - x_robot_next)**2 + 
                        (obs_future_pos[1] - y_robot_next)**2
                    )
                    if dist < r_robot + r_obs:
                        feasible = False
                        break
                
                if not feasible:
                    continue
                
                feasible_count += 1
                
                # Skip zero velocity
                if v == 0 and omega == 0:
                    continue
                
                # Compute six-term objective function (Eq. 6)
                j_heading = compute_j_heading(robot_state, goal, alpha_h)
                j_distance = compute_j_distance(robot_state, goal, alpha_d)
                j_clearance = compute_j_clearance(
                    robot_state, control, dt, obstacle_current, d_safe, alpha_c, dynamics
                )
                j_future_obs = compute_j_future_obs(
                    robot_state, control, dt, obstacle_current, obstacle_previous,
                    d_safe, alpha_f, dynamics
                )
                j_risk_obs = compute_j_risk_obs(
                    robot_state, obstacle_current, d_safe,
                    beta_1, beta_2, beta_3, beta_4, beta_5
                )
                j_count_obs = compute_j_count_obs(robot_state, obstacle_current)
                
                # Sum all terms (Eq. 6)
                j = j_heading + j_distance + j_clearance + j_future_obs + j_risk_obs + j_count_obs
                
                # Select argmax (Eq. 13)
                if j > best_j:
                    best_j = j
                    best_v = v
                    best_omega = omega
        
        # Check if any feasible velocity was found
        if feasible_count == 0:
            # No feasible velocity - stop robot
            best_v = 0.0
            best_omega = 0.0
        
        # Update previous obstacle positions for next iteration (Environment is static, so this is a no-op for two_rooms_simple)
        # For dynamic environments, obstacle_current would be updated externally; we just track it internally
        obstacle_previous = [obs.clone() for obs in obstacle_current]
        
        # Apply selected control and update robot state (Eq. 14)
        control = torch.tensor([best_v, best_omega], dtype=torch.float64)
        robot_next = dynamics.step(robot_state, control, dt)
        
        # Record trajectory
        trajectory.append((robot_state.clone(), control.clone()))
        
        # Update robot state
        robot_state = robot_next
        
        iteration += 1
    
    # Determine status
    if converged:
        status = "success"
    elif feasible_count == 0 and iteration > 0:
        status = "infeasible"
    else:
        status = "timeout"
    
    # Compile statistics
    stats = {
        "iterations": iteration,
        "final_distance_to_goal": dist_to_goal.item(),
        "trajectory_length": len(trajectory),
    }
    
    return PlanResult(
        trajectory=trajectory if trajectory else None,
        status=status,
        stats=stats
    )
