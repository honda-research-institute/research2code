"""
Model definitions for PDWA (Predictive Dynamic Window Approach).

This module implements the robot dynamics and collision model for the
Enhanced DWA with Dynamic Obstacle Behavior Prediction method from:
"Enhancing Obstacle Avoidance in Dynamic Window Approach via Dynamic Obstacle Behavior Prediction"
by Bongsu Hahn, Actuators 2025.

The robot is modeled as a unicycle (two-wheeled mobile robot) with unicycle
kinematics. The collision model uses circular robot and obstacle bodies for
distance-based collision checking.

# paper-element: concept-robot-model
# paper-element: eq-robot-kinematics
"""

from __future__ import annotations

import torch
from torch import Tensor


# essential: Unicycle kinematics with Euler integration
class UnicycleDynamics:
    """Unicycle kinematic model for a two-wheeled mobile robot.
    
    State: (x, y, theta) in R^3
    Control: (v, omega) in R^2
    
    Implements Euler integration of the unicycle kinematic equations:
        dx/dt = v * cos(theta)
        dy/dt = v * sin(theta)
        dtheta/dt = omega
    
    # paper-element: eq-robot-kinematics
    # paper-element: eq-robot-position-update
    """
    
    def __init__(self, V_max: float = 1.5, omega_max: float = 1.0):
        """Initialize unicycle dynamics with velocity bounds.
        
        Args:
            V_max: Maximum linear velocity (m/s). Paper value: 1.5 m/s.
            omega_max: Maximum angular velocity (rad/s). Paper value: 1.0 rad/s.
        """
        self.V_max = V_max
        self.omega_max = omega_max
    
    def step(self, state: Tensor, control: Tensor, dt: float) -> Tensor:
        """Compute next state using Euler integration.
        
        Args:
            state: Current state (x, y, theta) with shape (3,)
            control: Control input (v, omega) with shape (2,)
            dt: Time step in seconds
            
        Returns:
            Next state (x', y', theta') with shape (3,)
            
        # paper-element: eq-robot-position-update
        """
        x, y, theta = state[0], state[1], state[2]
        v, omega = control[0], control[1]
        
        # Euler integration of unicycle kinematics
        x_next = x + v * torch.cos(theta) * dt
        y_next = y + v * torch.sin(theta) * dt
        theta_next = theta + omega * dt
        
        return torch.stack([x_next, y_next, theta_next])
    
    def step_jacobian(self, state: Tensor, control: Tensor, dt: float) -> tuple[Tensor, Tensor]:
        """Compute Jacobians of the step function.
        
        Returns:
            Tuple of (df/dx, df/du) where:
                df/dx: 3x3 Jacobian of next state w.r.t. current state
                df/du: 3x2 Jacobian of next state w.r.t. control input
        """
        x, y, theta = state[0], state[1], state[2]
        v, omega = control[0], control[1]
        
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        # df/dx: Jacobian w.r.t. state (x, y, theta)
        # d(x_next)/dx = 1, d(x_next)/dtheta = -v*sin(theta)*dt
        # d(y_next)/dx = 0, d(y_next)/dtheta = v*cos(theta)*dt
        # d(theta_next)/dx = 0, d(theta_next)/dtheta = 1
        df_dx = torch.tensor([
            [1.0, 0.0, -v * sin_theta * dt],
            [0.0, 1.0, v * cos_theta * dt],
            [0.0, 0.0, 1.0]
        ], dtype=state.dtype, device=state.device)
        
        # df/du: Jacobian w.r.t. control (v, omega)
        # d(x_next)/dv = cos(theta)*dt, d(x_next)/domega = 0
        # d(y_next)/dv = sin(theta)*dt, d(y_next)/domega = 0
        # d(theta_next)/dv = 0, d(theta_next)/domega = dt
        df_du = torch.tensor([
            [cos_theta * dt, 0.0],
            [sin_theta * dt, 0.0],
            [0.0, dt]
        ], dtype=state.dtype, device=state.device)
        
        return df_dx, df_du


# essential: Circular robot body for collision checking
class CollisionModel:
    """Collision model for circular robot and obstacles.
    
    Uses distance-based collision checking: collision occurs when
    distance between robot center and obstacle center is less than
    r_robot + r_obs.
    
    # paper-element: concept-obstacle-model
    """
    
    def __init__(self, r_robot: float = 0.1, r_obs: float = 0.15):
        """Initialize collision model with robot and obstacle radii.
        
        Args:
            r_robot: Robot radius in meters. Paper value: 0.1 m.
            r_obs: Obstacle radius in meters. Paper value: 0.15 m.
        """
        self.r_robot = r_robot
        self.r_obs = r_obs
        self.collision_threshold = r_robot + r_obs
    
    def is_in_collision(self, state: Tensor, obstacle_positions: list[Tensor]) -> bool:
        """Check if robot state is in collision with any obstacle.
        
        Args:
            state: Robot state (x, y, theta) with shape (3,)
            obstacle_positions: List of obstacle positions, each (x, y) with shape (2,)
            
        Returns:
            True if robot is in collision with any obstacle, False otherwise.
        """
        x, y = state[0], state[1]
        robot_pos = torch.tensor([x, y], dtype=state.dtype, device=state.device)
        
        for obs_pos in obstacle_positions:
            distance = torch.norm(robot_pos - obs_pos)
            if distance < self.collision_threshold:
                return True
        
        return False
    
    def distance_to_obstacle(self, state: Tensor, obstacle_positions: list[Tensor]) -> float:
        """Compute minimum distance from robot to any obstacle.
        
        Args:
            state: Robot state (x, y, theta) with shape (3,)
            obstacle_positions: List of obstacle positions, each (x, y) with shape (2,)
            
        Returns:
            Minimum distance from robot center to obstacle center minus collision_threshold.
            Positive value means no collision; negative or zero means collision.
        """
        x, y = state[0], state[1]
        robot_pos = torch.tensor([x, y], dtype=state.dtype, device=state.device)
        
        min_distance = float('inf')
        for obs_pos in obstacle_positions:
            distance = torch.norm(robot_pos - obs_pos)
            min_distance = min(min_distance, distance.item())
        
        return min_distance - self.collision_threshold
