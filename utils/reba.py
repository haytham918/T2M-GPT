"""
Purpose: This script contains functions to calculate the differentiable REBA score for different body parts using PyTorch.
Reference: https://ergo-plus.com/reba-assessment-tool-guide/
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from jupyterlab.utils import deprecated
from mpl_toolkits.mplot3d import Axes3D

table_a_coordinates = [
        # Neck, Leg, Trunk, Score A
        [1, 1, 1, 1], [1, 1, 2, 2], [1, 1, 3, 2], [1, 1, 4, 3], [1, 1, 5, 4],
        [1, 2, 1, 2], [1, 2, 2, 3], [1, 2, 3, 4], [1, 2, 4, 5], [1, 2, 5, 6],
        [1, 3, 1, 3], [1, 3, 2, 4], [1, 3, 3, 5], [1, 3, 4, 6], [1, 3, 5, 7],
        [1, 4, 1, 4], [1, 4, 2, 5], [1, 4, 3, 6], [1, 4, 4, 7], [1, 4, 5, 8],
        [2, 1, 1, 1], [2, 1, 2, 3], [2, 1, 3, 4], [2, 1, 4, 5], [2, 1, 5, 6],
        [2, 2, 1, 2], [2, 2, 2, 4], [2, 2, 3, 5], [2, 2, 4, 6], [2, 2, 5, 7],
        [2, 3, 1, 3], [2, 3, 2, 5], [2, 3, 3, 6], [2, 3, 4, 7], [2, 3, 5, 8],
        [2, 4, 1, 4], [2, 4, 2, 6], [2, 4, 3, 7], [2, 4, 4, 8], [2, 4, 5, 9],
        [3, 1, 1, 3], [3, 1, 2, 4], [3, 1, 3, 5], [3, 1, 4, 6], [3, 1, 5, 7],
        [3, 2, 1, 3], [3, 2, 2, 5], [3, 2, 3, 6], [3, 2, 4, 7], [3, 2, 5, 8],
        [3, 3, 1, 5], [3, 3, 2, 6], [3, 3, 3, 7], [3, 3, 4, 8], [3, 3, 5, 9],
        [3, 4, 1, 6], [3, 4, 2, 7], [3, 4, 3, 8], [3, 4, 4, 9], [3, 4, 5, 9],
    ]

table_b_coordinates = [
        # Lower Arm, Wrist, Upper Arm, Score B
        [1, 1, 1, 1], [1, 1, 2, 1], [1, 1, 3, 3], [1, 1, 4, 4], [1, 1, 5, 6], [1, 1, 6, 7],
        [1, 2, 1, 2], [1, 2, 2, 2], [1, 2, 3, 4], [1, 2, 4, 5], [1, 2, 5, 7], [1, 2, 6, 8],
        [1, 3, 1, 2], [1, 3, 2, 3], [1, 3, 3, 5], [1, 3, 4, 5], [1, 3, 5, 8], [1, 3, 6, 8],
        [2, 1, 1, 1], [2, 1, 2, 2], [2, 1, 3, 4], [2, 1, 4, 5], [2, 1, 5, 7], [2, 1, 6, 8],
        [2, 2, 1, 2], [2, 2, 2, 3], [2, 2, 3, 5], [2, 2, 4, 6], [2, 2, 5, 8], [2, 2, 6, 9],
        [2, 3, 1, 3], [2, 3, 2, 4], [2, 3, 3, 5], [2, 3, 4, 7], [2, 3, 5, 8], [2, 3, 6, 9],
    ]

table_c_coordinates = [
        # Score A, Score B, Score C
        [1, 1, 1], [1, 2, 1], [1, 3, 1], [1, 4, 2], [1, 5, 3], [1, 6, 3], [1, 7, 4], [1, 8, 5], [1, 9, 6], [1, 10, 7], [1, 11, 7], [1, 12, 7],
        [2, 1, 1], [2, 2, 2], [2, 3, 2], [2, 4, 3], [2, 5, 4], [2, 6, 4], [2, 7, 5], [2, 8, 6], [2, 9, 6], [2, 10, 7], [2, 11, 7], [2, 12, 8],
        [3, 1, 2], [3, 2, 3], [3, 3, 3], [3, 4, 3], [3, 5, 4], [3, 6, 5], [3, 7, 6], [3, 8, 7], [3, 9, 7], [3, 10, 8], [3, 11, 8], [3, 12, 8],
        [4, 1, 3], [4, 2, 4], [4, 3, 4], [4, 4, 4], [4, 5, 5], [4, 6, 6], [4, 7, 7], [4, 8, 8], [4, 9, 8], [4, 10, 9], [4, 11, 9], [4, 12, 9],
        [5, 1, 4], [5, 2, 4], [5, 3, 4], [5, 4, 5], [5, 5, 6], [5, 6, 7], [5, 7, 8], [5, 8, 9], [5, 9, 9], [5, 10, 9], [5, 11, 9], [5, 12, 9],
        [6, 1, 6], [6, 2, 6], [6, 3, 6], [6, 4, 7], [6, 5, 8], [6, 6, 8], [6, 7, 9], [6, 8, 9], [6, 9, 10], [6, 10, 10], [6, 11, 10], [6, 12, 10],
        [7, 1, 7], [7, 2, 7], [7, 3, 7], [7, 4, 8], [7, 5, 9], [7, 6, 9], [7, 7, 9], [7, 8, 10], [7, 9, 10], [7, 10, 11], [7, 11, 11], [7, 12, 11],
        [8, 1, 8], [8, 2, 8], [8, 3, 8], [8, 4, 9], [8, 5, 10], [8, 6, 10], [8, 7, 10], [8, 8, 10], [8, 9, 10], [8, 10, 11], [8, 11, 11], [8, 12, 11],
        [9, 1, 9], [9, 2, 9], [9, 3, 9], [9, 4, 10], [9, 5, 10], [9, 6, 10], [9, 7, 11], [9, 8, 11], [9, 9, 11], [9, 10, 12], [9, 11, 12], [9, 12, 12],
        [10, 1, 10], [10, 2, 10], [10, 3, 10], [10, 4, 11], [10, 5, 11], [10, 6, 11], [10, 7, 11], [10, 8, 12], [10, 9, 12], [10, 10, 12], [10, 11, 12], [10, 12, 12],
        [11, 1, 11], [11, 2, 11], [11, 3, 11], [11, 4, 11], [11, 5, 12], [11, 6, 12], [11, 7, 12], [11, 8, 12], [11, 9, 12], [11, 10, 12], [11, 11, 12], [11, 12, 12],
        [12, 1, 12], [12, 2, 12], [12, 3, 12], [12, 4, 12], [12, 5, 12], [12, 6, 12], [12, 7, 12], [12, 8, 12], [12, 9, 12], [12, 10, 12], [12, 11, 12], [12, 12, 12],

    ]

table_risk_coordinates = [[1,0], [2,1], [3,1], [4,2], [5,2], [6,2], [7,2], [8,3], [9,3], [10,3], [11,4], [12,4], [13,4], [14,4], [15,4]]

def get_neck_score(angle, steepness=10.0):
    """
    Calculate a fully differentiable REBA neck score based on the angle of the neck using PyTorch.
    
    Parameters:
        angle (torch.Tensor): The neck angle in degrees. Can be a single value or a tensor of values.
        steepness (float): The steepness of the sigmoid transitions. The lower the value, the smoother the transition.
    
    Returns:
        torch.Tensor: Differentiable REBA score(s).
    """
    # Define thresholds
    low_threshold = 0.0
    high_threshold = 20.0

    # Smooth transition for each range using sigmoid
    low_range = torch.sigmoid(-steepness * angle)  # Below 0
    high_range = torch.sigmoid(steepness * (angle - high_threshold))  # Above 20

    # Weighted sum to ensure smoothness
    score = 1 + low_range + high_range
    return score

def get_trunk_score(angle, steepness=10.0):
    """
    Calculate a fully differentiable REBA trunk score based on the trunk angle using PyTorch.
    
    Parameters:
        angle (torch.Tensor): The trunk angle in degrees. Can be a single value or a tensor of values.
        steepness (float): The steepness of the sigmoid transitions.
    
    Returns:
        torch.Tensor: Differentiable REBA score(s).
    """
    # Define thresholds
    low_threshold_1 = -20.0
    high_threshold_1 = 20.0
    high_threshold_2 = 60.0

    # Smooth transition for each range using sigmoid
    below_low = torch.sigmoid(-steepness * (angle - low_threshold_1))  # Smooth transition below -20
    in_low_range = torch.sigmoid(steepness * (angle - low_threshold_1)) * (1 - torch.sigmoid(steepness * (angle - high_threshold_1)))  # -20 to 20
    in_mid_range = torch.sigmoid(steepness * (angle - high_threshold_1)) * (1 - torch.sigmoid(steepness * (angle - high_threshold_2)))  # 20 to 60
    above_high = torch.sigmoid(steepness * (angle - high_threshold_2))  # Smooth transition above 60

    # Combine scores
    score = 1 + below_low * 2 + in_low_range * 1 + in_mid_range * 2 + above_high * 3

    return score

def get_leg_score(angle, foot_lifted=0, steepness=10.0):
    """
    Calculate a differentiable REBA leg score based on the knee angle and foot lifted status using PyTorch.
    
    Parameters:
        angle (torch.Tensor): The knee angle in degrees. Can be a single value or a tensor of values.
        foot_lifted (torch.Tensor): Binary tensor (0 or 1), where 1 indicates the foot is lifted.
        steepness (float): The steepness of the sigmoid transitions.
    
    Returns:
        torch.Tensor: Differentiable REBA score(s).
    """
    # Define knee angle thresholds
    low_threshold = 30.0
    high_threshold = 60.0

    # Smooth transition for each knee angle range
    in_mid_range = torch.sigmoid(steepness * (angle - low_threshold)) * (1 - torch.sigmoid(steepness * (angle - high_threshold)))  # 30 <= knee_angle <= 60
    above_high = torch.sigmoid(steepness * (angle - high_threshold))  # knee_angle > 60

    # Foot lifted contributes 2 if lifted, 1 if not
    foot_score = 1 + foot_lifted  # 1 if not lifted, 2 if lifted

    # Base score
    score = foot_score + in_mid_range + 2 * above_high

    return score

def get_upper_arm_score(angle, steepness=10.0):
    """
    Calculate a differentiable score based on the upper arm angle using PyTorch.
    
    Parameters:
        angle (torch.Tensor): The upper arm angle in degrees. Can be a single value or a tensor of values.
        steepness (float): The steepness of the sigmoid transitions. Higher values result in sharper transitions.
    
    Returns:
        torch.Tensor: Differentiable upper arm score.
    """
    # Define thresholds
    low_threshold_1 = -20.0
    high_threshold_1 = 20.0
    high_threshold_2 = 45.0
    high_threshold_3 = 90.0

    # Smooth transitions using sigmoid
    below_low = torch.sigmoid(-steepness * (angle - low_threshold_1))  # Active when angle < -20
    in_mid_low = torch.sigmoid(steepness * (angle - low_threshold_1)) * (1 - torch.sigmoid(steepness * (angle - high_threshold_1)))  # -20 <= angle <= 20
    in_mid_high = torch.sigmoid(steepness * (angle - high_threshold_1)) * (1 - torch.sigmoid(steepness * (angle - high_threshold_2)))  # 20 < angle <= 45
    in_high = torch.sigmoid(steepness * (angle - high_threshold_2)) * (1 - torch.sigmoid(steepness * (angle - high_threshold_3)))  # 45 < angle <= 90
    above_high = torch.sigmoid(steepness * (angle - high_threshold_3))  # Active when angle > 90

    # Calculate the score
    score = 1 * in_mid_low + 2 * (below_low + in_mid_high) + 3 * in_high + 4 * above_high

    return score

def get_lower_arm_score(angle, steepness=10.0):
    """
    Calculate a differentiable score based on the lower arm angle using PyTorch.
    
    Parameters:
        angle (torch.Tensor): The lower arm angle in degrees. Can be a single value or a tensor of values.
        steepness (float): The steepness of the sigmoid transitions. Higher values result in sharper transitions.
    
    Returns:
        torch.Tensor: Differentiable lower arm score.
    """
    # Define thresholds
    low_threshold = 60.0
    high_threshold = 100.0

    # Smooth transitions using sigmoid
    below_low = torch.sigmoid(-steepness * (angle - low_threshold))  # Active when angle < 60
    in_range = torch.sigmoid(steepness * (angle - low_threshold)) * (1 - torch.sigmoid(steepness * (angle - high_threshold)))  # 60 <= angle <= 100
    above_high = torch.sigmoid(steepness * (angle - high_threshold))  # Active when angle > 100

    # Calculate the score
    score = 1 * in_range + 2 * (below_low + above_high)

    return score

def get_wrist_score(angle, steepness=10.0):
    """
    Calculate a differentiable score based on the wrist angle using PyTorch.
    
    Parameters:
        angle (torch.Tensor): The wrist angle in degrees. Can be a single value or a tensor of values.
        steepness (float): The steepness of the sigmoid transitions. Higher values result in sharper transitions.
    
    Returns:
        torch.Tensor: Differentiable wrist angle score.
    """
    # Define thresholds
    low_threshold = -15.0
    high_threshold = 15.0

    # Smooth transitions using sigmoid
    below_low = torch.sigmoid(-steepness * (angle - low_threshold))  # Active when angle < -15
    in_range = torch.sigmoid(steepness * (angle - low_threshold)) * (1 - torch.sigmoid(steepness * (angle - high_threshold)))  # -15 <= angle <= 15
    above_high = torch.sigmoid(steepness * (angle - high_threshold))  # Active when angle > 15

    # Calculate the score
    score = 1 * in_range + 2 * (below_low + above_high)

    return score

def get_score_a(neck, leg, trunk, gamma=10.0):
    """
    Calculate a differentiable Score A based on Neck, Leg, and Trunk floating scores using PyTorch.
    
    Parameters:
        neck (torch.Tensor): Neck floating scores.
        leg (torch.Tensor): Leg floating scores.
        trunk (torch.Tensor): Trunk floating scores.
        gamma (float): RBF kernel width (controls smoothness).

    Returns:
        torch.Tensor: Differentiable Score A.
    """
    table_coordinates = table_a_coordinates
    coordinates = torch.tensor([coord[:3] for coord in table_coordinates], dtype=torch.float32)  # (N, 3)
    scores = torch.tensor([coord[3] for coord in table_coordinates], dtype=torch.float32)  # (N,)
    
    # Ensure inputs are tensors
    neck = torch.tensor(neck) if not isinstance(neck, torch.Tensor) else neck
    leg = torch.tensor(leg) if not isinstance(leg, torch.Tensor) else leg
    trunk = torch.tensor(trunk) if not isinstance(trunk, torch.Tensor) else trunk
    
    # Stack input values into a single tensor (1, 3)
    inputs = torch.stack([neck, leg, trunk], dim=-1)  # Shape: (batch_size, 3)
    
    # Compute RBF distances
    distances = torch.cdist(inputs, coordinates.unsqueeze(0), p=2)  # Shape: (batch_size, N)
    weights = torch.exp(-gamma * distances**2)  # RBF weights (Gaussian kernel)
    
    # Normalize weights
    weights = weights / weights.sum(dim=-1, keepdim=True)
    
    # Compute the interpolated Score A
    score_a = torch.matmul(weights, scores)  # Weighted sum of scores
    
    return score_a.squeeze()

def plot_score_a_3d(neck_value, gamma=10.0):
    """
    Plot Score A as a 3D surface with leg and trunk as axes, and neck fixed.
    
    Parameters:
        neck_value (float): Fixed neck value.
        gamma (float): RBF kernel width (controls smoothness).
    """
    table_coordinates = table_a_coordinates
    coordinates = torch.tensor([coord[:3] for coord in table_coordinates], dtype=torch.float32)  # (N, 3)
    scores = torch.tensor([coord[3] for coord in table_coordinates], dtype=torch.float32)  # (N,)
    predicted_scores = get_score_a(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], gamma=gamma)
    for i, (coord, score, pred_score) in enumerate(zip(coordinates, scores, predicted_scores)):
        print(f"Table {i + 1}: {coord} -> Score: {score:.2f} | Predicted Score: {pred_score:.2f}")
        
    # Generate grid for leg and trunk
    leg_values = torch.linspace(1.0, 5.0, 50)  # Leg scores from 1 to 5
    trunk_values = torch.linspace(1.0, 5.0, 50)  # Trunk scores from 1 to 5
    leg_grid, trunk_grid = torch.meshgrid(leg_values, trunk_values, indexing="ij")

    # Fixed neck value
    neck = torch.tensor([neck_value])

    # Flatten grid for batch processing
    leg_flat = leg_grid.flatten()
    trunk_flat = trunk_grid.flatten()

    # Compute Score A for each point in the grid
    score_a = get_score_a(neck.expand_as(leg_flat), leg_flat, trunk_flat, gamma=gamma)

    # Reshape Score A back to grid shape
    score_a_grid = score_a.reshape(leg_grid.shape)

    # Convert to numpy for plotting
    leg_np = leg_grid.numpy()
    trunk_np = trunk_grid.numpy()
    score_a_np = score_a_grid.detach().numpy()

    # Plot 3D surface
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(leg_np, trunk_np, score_a_np, cmap="viridis", edgecolor="none")

    # Labels and title
    ax.set_xlabel("Leg Score")
    ax.set_ylabel("Trunk Score")
    ax.set_zlabel("Score A")
    ax.set_title(f"3D Plot of Score A (Neck = {neck_value})")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    plt.show()

def get_score_b(upper_arm, lower_arm, wrist, gamma=10.0):
    """
    Calculate a differentiable Score B based on Upper Arm, Lower Arm, and Wrist floating scores using PyTorch.
    
    Parameters:
        upper_arm (torch.Tensor): Upper Arm floating scores.
        lower_arm (torch.Tensor): Lower Arm floating scores.
        wrist (torch.Tensor): Wristrist floating scores.
        gamma (float): RBF kernel width (controls smoothness). The lower the value, the smoother the interpolation.

    Returns:
        torch.Tensor: Differentiable Score B.
    """
    table_coordinates = table_b_coordinates
    coordinates = torch.tensor([coord[:3] for coord in table_coordinates], dtype=torch.float32)  # (N, 3)
    scores = torch.tensor([coord[3] for coord in table_coordinates], dtype=torch.float32)  # (N,)
    
    # Ensure inputs are tensors
    upper_arm = torch.tensor(upper_arm) if not isinstance(upper_arm, torch.Tensor) else upper_arm
    lower_arm = torch.tensor(lower_arm) if not isinstance(lower_arm, torch.Tensor) else lower_arm
    wrist = torch.tensor(wrist) if not isinstance(wrist, torch.Tensor) else wrist
    
    # Stack input values into a single tensor (1, 3)
    inputs = torch.stack([lower_arm, wrist, upper_arm], dim=-1)  # Shape: (batch_size, 3)
    
    # Compute RBF distances
    distances = torch.cdist(inputs, coordinates.unsqueeze(0), p=2)  # Shape: (batch_size, N)
    weights = torch.exp(-gamma * distances**2)  # RBF weights (Gaussian kernel)
    
    # Normalize weights
    weights = weights / weights.sum(dim=-1, keepdim=True)
    
    # Compute the interpolated Score B
    score_b = torch.matmul(weights, scores)  # Weighted sum of scores
    
    return score_b.squeeze()

def plot_score_b_3d(lower_arm_value, gamma=10.0):
    """
    Plot Score B as a 3D surface with Upper Arm and Wrist as axes, and Lower Arm fixed.
    
    Parameters:
        lower_arm_value (float): Fixed Lower Arm value.
        gamma (float): RBF kernel width (controls smoothness).
    """
    table_coordinates = table_b_coordinates
    coordinates = torch.tensor([coord[:3] for coord in table_coordinates], dtype=torch.float32)  # (N, 3)
    scores = torch.tensor([coord[3] for coord in table_coordinates], dtype=torch.float32)  # (N,)
    predicted_scores = get_score_b(coordinates[:, 2], coordinates[:, 0], coordinates[:, 1], gamma=gamma)
    for i, (coord, score, pred_score) in enumerate(zip(coordinates, scores, predicted_scores)):
        print(f"Table {i + 1}: {coord} -> Score: {score:.2f} | Predicted Score: {pred_score:.2f}")
        
    # Generate grid for upper_arm and wrist
    upper_arm_values = torch.linspace(1.0, 6.0, 50)  # Upper Arm scores from 1 to 6
    wrist_values = torch.linspace(1.0, 3.0, 50)  # Wrist scores from 1 to 3
    upper_arm_grid, wrist_grid = torch.meshgrid(upper_arm_values, wrist_values, indexing="ij")

    # Fixed lower_arm value
    lower_arm = torch.tensor([lower_arm_value])

    # Flatten grid for batch processing
    upper_arm_flat = upper_arm_grid.flatten()
    wrist_flat = wrist_grid.flatten()

    # Compute Score B for each point in the grid
    score_b = get_score_b(upper_arm_flat, lower_arm.expand_as(upper_arm_flat), wrist_flat, gamma=gamma)

    # Reshape Score B back to grid shape
    score_b_grid = score_b.reshape(upper_arm_grid.shape)

    # Convert to numpy for plotting
    upper_arm_np = upper_arm_grid.numpy()
    wrist_np = wrist_grid.numpy()
    score_b_np = score_b_grid.detach().numpy()

    # Plot 3D surface
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(upper_arm_np, wrist_np, score_b_np, cmap="viridis", edgecolor="none")

    # Labels and title
    ax.set_xlabel("Upper_Arm Score")
    ax.set_ylabel("Wrist Score")
    ax.set_zlabel("Score B")
    ax.set_title(f"3D Plot of Score B (Lower Arm = {lower_arm_value})")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    plt.show()

def get_score_c(score_a, score_b, gamma=10.0):
    """
    Calculate a differentiable Score C based on Score A and Score B floating scores using PyTorch.
    
    Parameters:
        score_a (torch.Tensor): Score A floating scores.
        score_b (torch.Tensor): Score B floating scores.
        gamma (float): RBF kernel width (controls smoothness). The lower the value, the smoother the interpolation.

    Returns:
        torch.Tensor: Differentiable Score C.
    """
    table_coordinates = table_c_coordinates
    coordinates = torch.tensor([coord[:2] for coord in table_coordinates], dtype=torch.float32)  # (N, 2)
    scores = torch.tensor([coord[2] for coord in table_coordinates], dtype=torch.float32)  # (N,)
    
    # Ensure inputs are tensors
    score_a = torch.tensor(score_a) if not isinstance(score_a, torch.Tensor) else score_a
    score_b = torch.tensor(score_b) if not isinstance(score_b, torch.Tensor) else score_b

    # Stack input values into a single tensor (1, 2)
    inputs = torch.stack([score_a, score_b], dim=-1)  # Shape: (batch_size, 2)
    
    # Compute RBF distances
    distances = torch.cdist(inputs, coordinates.unsqueeze(0), p=2)  # Shape: (batch_size, N)
    weights = torch.exp(-gamma * distances**2)  # RBF weights (Gaussian kernel)
    
    # Normalize weights
    weights = weights / weights.sum(dim=-1, keepdim=True)
    
    # Compute the interpolated Score A
    score_c = torch.matmul(weights, scores)  # Weighted sum of scores
    
    return score_c.squeeze()

def plot_score_c_3d(gamma=10.0):
    """
    Plot Score C as a 3D surface with Score A and Score B as axes.
    
    Parameters:
        gamma (float): RBF kernel width (controls smoothness).
    """
    table_coordinates = table_c_coordinates
    coordinates = torch.tensor([coord[:2] for coord in table_coordinates], dtype=torch.float32)  # (N, 3)
    scores = torch.tensor([coord[2] for coord in table_coordinates], dtype=torch.float32)  # (N,)
    predicted_scores = get_score_c(coordinates[:, 0], coordinates[:, 1], gamma=gamma)
    for i, (coord, score, pred_score) in enumerate(zip(coordinates, scores, predicted_scores)):
        print(f"Table {i + 1}: {coord} -> Score: {score:.2f} | Predicted Score: {pred_score:.2f}")
        
    # Generate grid for leg and trunk
    score_a_values = torch.linspace(1.0, 12.0, 500)  # Upper Arm scores from 1 to 6
    score_b_values = torch.linspace(1.0, 12.0, 500)  # Wrist scores from 1 to 3
    score_a_grid, score_b_grid = torch.meshgrid(score_a_values, score_b_values, indexing="ij")

    # Flatten grid for batch processing
    score_a_flat = score_a_grid.flatten()
    score_b_flat = score_b_grid.flatten()

    # Compute Score C for each point in the grid
    score_c = get_score_c(score_a_flat, score_b_flat, gamma=gamma)

    # Reshape Score C back to grid shape
    score_c_grid = score_c.reshape(score_a_grid.shape)

    # Convert to numpy for plotting
    score_a_np = score_a_grid.numpy()
    score_b_np = score_b_grid.numpy()
    score_c_np = score_c_grid.detach().numpy()

    # Plot 3D surface
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(score_a_np, score_b_np, score_c_np, cmap="viridis", edgecolor="none")

    # Labels and title
    ax.set_xlabel("Score A")
    ax.set_ylabel("Score B")
    ax.set_zlabel("Score C")
    ax.set_title(f"3D Plot of Score C")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    plt.show()

@deprecated
def get_action_level(score, gamma=10.0):
    """
    Get the action level based on the REBA score.
        1   negligible risk, no action required (0)
        2-3 low risk, change may be needed (1)
        4-7 medium risk, further investigation, change soon (2)
        8-10 high risk, investigate and implement change (3)
        11+ very high risk, implement change (4)
    Parameters:
        score (torch.Tensor): The REBA score, approximated by our differentiable score.

    Returns:
        (torch.Tensor): The risk level based on the REBA score.
    """
    # todo: think more, i think it should be RBF
    # Define thresholds
    table_coordinates = table_risk_coordinates
    coordinates = torch.tensor([coord[0] for coord in table_coordinates], dtype=torch.float32)  # (N,)
    action_levels = torch.tensor([coord[1] for coord in table_coordinates], dtype=torch.float32)  # (N,)

    # Ensure inputs are tensors
    score = torch.tensor(score) if not isinstance(score, torch.Tensor) else score

    # Stack input values into a single tensor (1, 1)
    inputs = torch.stack([score], dim=-1)  # Shape: (batch_size, 1)

    # Compute RBF distances
    distances = torch.cdist(inputs, coordinates.unsqueeze(0), p=2)  # Shape: (batch_size, N)
    weights = torch.exp(-gamma * distances ** 2)  # RBF weights (Gaussian kernel)

    # Normalize weights
    weights = weights / weights.sum(dim=-1, keepdim=True)

    # Compute the interpolated Score A
    this_action_level = torch.matmul(weights, action_levels)  # Weighted sum of scores

    return this_action_level.squeeze()

@deprecated
def plot_action_level(gamma=10.0):
    """

    """
    table_coordinates = table_risk_coordinates
    coordinates = torch.tensor([coord[0] for coord in table_coordinates], dtype=torch.float32)  # (N,)
    action_levels = torch.tensor([coord[1] for coord in table_coordinates], dtype=torch.float32)  # (N,)

    predicted_action_levels = get_action_level(coordinates[:, 0], gamma=gamma)
    for i, (coord, score, pred_score) in enumerate(zip(coordinates, action_levels, predicted_action_levels)):
        print(f"Table {i + 1}: {coord} -> action_levels: {score:.2f} | Predicted action_levels: {pred_score:.2f}")

    # Generate grid for leg and trunk
    score_c_values = torch.linspace(1.0, 15.0, 100)
    # Flatten grid for batch processing
    score_c_values = score_c_values.flatten()

    # Compute Score C for each point in the grid
    this_action_level = get_score_c(score_c_values, gamma=gamma)

    # Convert to numpy for plotting
    this_action_level_np = this_action_level.detach().numpy()

    # Plot 3D surface
    # Plot the graph
