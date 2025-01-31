import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))

import reba
from utils.motion_process import recover_from_ric


class ErgoLoss(nn.Module):
    def __init__(self, nb_joints):
        super(ErgoLoss, self).__init__()

        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.nb_joints = nb_joints  # 22 for humanML3D, 21 for KIT (not using)

    def anlge_3D(self, v1, v2):
        # todo: write test
        """
        Compute the angle between two vectors in 3D space
        :param v1: torch.tensor, shape (batch_size, 3)
        :param v2: torch.tensor, shape (batch_size, 3)
        :return: torch.tensor, shape (batch_size)
        """
        dot_product = torch.sum(v1 * v2, dim=-1)
        norm_v1 = torch.norm(v1, dim=-1)
        norm_v2 = torch.norm(v2, dim=-1)

        # Avoid division by zero
        eps = 1e-8
        norm_v1 = torch.clamp(norm_v1, min=eps)
        norm_v2 = torch.clamp(norm_v2, min=eps)

        cos_angle = torch.clamp(dot_product / (norm_v1 * norm_v2), -1.0 + eps, 1.0 - eps)
        angle = torch.acos(cos_angle)
        return angle

    def in_front_plane(self, pt_ref, pt1, pt2, pt3):
        # todo: write test
        """
        Check if a point is in front of a plane defined by three points
        :param pt_ref: torch.tensor, shape (batch_size, 3)
        :param pt1: torch.tensor, shape (batch_size, 3)
        :param pt2: torch.tensor, shape (batch_size, 3)
        :param pt3: torch.tensor, shape (batch_size, 3)
        :return: torch.tensor, shape (batch_size)
        """
        v1 = pt2 - pt1
        v2 = pt3 - pt1
        normal = torch.cross(v1, v2)
        ref_to_pt1 = pt1 - pt_ref
        dot_product = torch.sum(normal * ref_to_pt1, dim=-1)
        return dot_product > 0

    def forward(self, motion_pred):
        # Step 1: compute angles
        pred_xyz = recover_from_ric((motion_pred).float(), self.nb_joints).reshape(1, -1, self.nb_joints, 3)
        batch_size = pred_xyz.shape[1]
        ### trunk
        trunk_vec = pred_xyz[0, :, 12, :] - pred_xyz[0, :, 0, :]  # Root-->Neck_base
        y_pos_vec = torch.tensor([0, 1, 0]).repeat(batch_size, 1).float().cuda()  # same shape, with y +
        trunk_angle = self.anlge_3D(trunk_vec, y_pos_vec)  # trunk angle + - is the same score

        ### upper_arm
        right_upper_arm_vec = pred_xyz[0, :, 19, :] - pred_xyz[0, :, 17, :]
        left_upper_arm_vec = pred_xyz[0, :, 18, :] - pred_xyz[0, :, 16, :]
        right_upper_arm_angle = np.pi - self.anlge_3D(right_upper_arm_vec, trunk_vec)
        left_upper_arm_angle = np.pi - self.anlge_3D(left_upper_arm_vec, trunk_vec)  # + - is the same score
        upper_arm_angle = torch.max(right_upper_arm_angle, left_upper_arm_angle)

        ### lower_arm
        right_lower_arm_vec = pred_xyz[0, :, 21, :] - pred_xyz[0, :, 19, :]
        left_lower_arm_vec = pred_xyz[0, :, 20, :] - pred_xyz[0, :, 18, :]
        right_lower_arm_angle = self.anlge_3D(right_lower_arm_vec, right_upper_arm_vec)
        left_lower_arm_angle = self.anlge_3D(left_lower_arm_vec, left_upper_arm_vec)
        lower_arm_angle = torch.max(right_lower_arm_angle, left_lower_arm_angle)

        ### wrist: no wrist joint in the 22 kpts
        ### neck: no good kpts, head base (15) is too forward

        ### leg
        right_upper_leg_vec = pred_xyz[0, :, 5, :] - pred_xyz[0, :, 2, :]
        left_upper_leg_vec = pred_xyz[0, :, 4, :] - pred_xyz[0, :, 1, :]
        right_lower_leg_vec = pred_xyz[0, :, 8, :] - pred_xyz[0, :, 5, :]
        left_lower_leg_vec = pred_xyz[0, :, 7, :] - pred_xyz[0, :, 4, :]

        right_knee_angle = self.anlge_3D(right_upper_leg_vec, right_lower_leg_vec)
        left_knee_angle = self.anlge_3D(left_upper_leg_vec, left_lower_leg_vec)
        knee_angle = torch.max(right_knee_angle, left_knee_angle)

        # Step 2: joint scores

        trunk = reba.get_trunk_score(trunk_angle, steepness=1)
        upper_arm = reba.get_upper_arm_score(upper_arm_angle, steepness=1)
        lower_arm = reba.get_lower_arm_score(lower_arm_angle, steepness=1)
        leg = reba.get_leg_score(knee_angle, steepness=1)

        wrist = torch.ones_like(trunk)  # wrist = reba.get_wrist_score(angles, steepness=1)
        neck = torch.ones_like(trunk)  # neck = reba.get_neck_score(angles, steepness=1)

        # Step 3: REBA score
        score_a = reba.get_score_a(neck, leg, trunk)  # Score A value between 1 and 12
        score_b = reba.get_score_b(upper_arm, lower_arm, wrist)  # Score B value between 1 and 12
        score_c = reba.get_score_c(score_a, score_b)

        # # Step 4: REBA action level
        # action_level = reba.get_action_level(score_c)

        # Step 5: Loss
        loss = (score_c-1) ** 2 / 15**2  # quadratic loss normalized between 0 and 1
        loss = torch.sum(loss)
        return loss




# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # Load the data
# file_path = '/Users/leyangwen/Downloads/new_joints/000006.npy'
# data = np.load(file_path)  # (199, 22, 3)
#
# # Specify the frame to plot
# frame_index = 0
# frame_data = data[frame_index]
#
# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot each joint and mark with index
# for i, joint in enumerate(frame_data):
#     ax.scatter(joint[0], joint[1], joint[2], label=f'Joint {i}')
#     ax.text(joint[0], joint[1], joint[2], f'{i}', size=10, zorder=1, color='k')
#
# # Set labels
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
# # Show plot
# # plt.legend()
# plt.show()
