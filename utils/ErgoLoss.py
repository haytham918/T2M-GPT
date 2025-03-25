import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))

import reba
import numpy as np
from utils.motion_process import recover_from_ric
from visualize.simplify_loc2rot import joints2smpl
from models.rotation2xyz import Rotation2xyz

from ergo3d.geometryPytorch import Point, Plane, CoordinateSystem3D, JointAngles

class ErgoLoss(nn.Module):
    def __init__(self, nb_joints, verbose=False):
        super(ErgoLoss, self).__init__()

        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.nb_joints = nb_joints  # 22 for humanML3D, 21 for KIT (not using)
        self.verbose = verbose
        self.version_2 = False
        self.REBA_improve = 0.5

    def anlge_3D(self, v1, v2, output_type='degree'):
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
        eps = 1e-7
        norm_v1 = torch.clamp(norm_v1, min=eps)
        norm_v2 = torch.clamp(norm_v2, min=eps)

        cos_angle = torch.clamp(dot_product / (norm_v1 * norm_v2), -1.0 + eps, 1.0 - eps)
        angle = torch.acos(cos_angle)

        # angles here should be [0, pi], otherwise, convert
        angle = torch.min(angle, np.pi - angle)
        if output_type == 'degree':
            angle = angle * 180 / np.pi
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

    def marker_vids(self):  # read xml
        marker_vids = {
            'smpl': {'HDTP': 411, 'REAR': 3941, 'LEAR': 517, 'MDFH': 335, 'C7': 829, 'C7_d': 1305, 'SS': 3171, 'T8': 3024, 'XP': 3077, 'RPSIS': 5246, 'RASIS': 6573, 'LPSIS': 1781, 'LASIS': 3156,
                     'RAP': 4721, 'RAP_b': 5283, 'RAP_f': 5354, 'LAP': 1238, 'LAP_b': 2888, 'LAP_f': 1317, 'RLE': 5090, 'RME': 5202, 'LLE': 1621, 'LME': 1732, 'RRS': 5573, 'RUS': 5568, 'LRS': 2112,
                     'LUS': 2108, 'RMCP2': 5595, 'RMCP5': 5636, 'LMCP2': 2135, 'LMCP5': 2290, 'RIC': 5257, 'RGT': 4927, 'LIC': 1794, 'LGT': 1454, 'RMFC': 4842, 'RLFC': 4540, 'LMFC': 1369,
                     'LLFC': 1054, 'RMM': 6832, 'RLM': 6728, 'LMM': 3432, 'LLM': 3327, 'RMTP1': 6739, 'RMTP5': 6745, 'LMTP1': 3337, 'LMTP5': 3344, 'RHEEL': 6786, 'LHEEL': 3387},
            'smplh': {'HDTP': 411, 'REAR': 3941, 'LEAR': 517, 'MDFH': 335, 'C7': 829, 'C7_d': 1305, 'SS': 3171, 'T8': 3024, 'XP': 3077, 'RPSIS': 5246, 'RASIS': 6573, 'LPSIS': 1781, 'LASIS': 3156,
                      'RAP': 4721, 'RAP_b': 5283, 'RAP_f': 5354, 'LAP': 1238, 'LAP_b': 2888, 'LAP_f': 1317, 'RLE': 5090, 'RME': 5202, 'LLE': 1621, 'LME': 1732, 'RRS': 5573, 'RUS': 5568, 'LRS': 2112,
                      'LUS': 2108, 'RMCP2': 5595, 'RMCP5': 5636, 'LMCP2': 2135, 'LMCP5': 2290, 'RIC': 5257, 'RGT': 4927, 'LIC': 1794, 'LGT': 1454, 'RMFC': 4842, 'RLFC': 4540, 'LMFC': 1369,
                      'LLFC': 1054, 'RMM': 6832, 'RLM': 6728, 'LMM': 3432, 'LLM': 3327, 'RMTP1': 6739, 'RMTP5': 6745, 'LMTP1': 3337, 'LMTP5': 3344, 'RHEEL': 6786, 'LHEEL': 3387},
            'smplx': {'HDTP': 9011, 'REAR': 1050, 'LEAR': 560, 'MDFH': 8949, 'C7': 3353, 'C7_d': 5349, 'SS': 5533, 'T8': 5495, 'XP': 5534, 'RPSIS': 7141, 'RASIS': 8421, 'LPSIS': 4405, 'LASIS': 5727,
                      'RAP': 6175, 'RAP_b': 6632, 'RAP_f': 7253, 'LAP': 3414, 'LAP_b': 4431, 'LAP_f': 4517, 'RLE': 6695, 'RME': 7107, 'LLE': 4251, 'LME': 4371, 'RRS': 7462, 'RUS': 7458, 'LRS': 4726,
                      'LUS': 4722, 'RMCP2': 7483, 'RMCP5': 7525, 'LMCP2': 4747, 'LMCP5': 4788, 'RIC': 7149, 'RGT': 6832, 'LIC': 4413, 'LGT': 4088, 'RMFC': 6747, 'RLFC': 6445, 'LMFC': 3999,
                      'LLFC': 3684, 'RMM': 8680, 'RLM': 8576, 'LMM': 8892, 'LLM': 5882, 'RMTP1': 8587, 'RMTP5': 8593, 'LMTP1': 5893, 'LMTP5': 5899, 'RHEEL': 8634, 'LHEEL': 8846}}
        return marker_vids


    def get_smpl_mesh(self, motions):
        frames = motions.shape[0]  # batch_size
        j2s = joints2smpl(num_frames=frames, device_id=0, cuda=True)
        rot2xyz = Rotation2xyz(device=torch.device("cuda:0"))
        # faces = rot2xyz.smpl_model.faces
        motion_tensor, opt_dict = j2s.joint2smpl(motions)  # [nframes, njoints, 3]

        vertices = rot2xyz(torch.tensor(motion_tensor).clone(), mask=None,
                           pose_rep='rot6d', translation=True, glob=True,
                           jointstype='vertices',
                           vertstrans=True)
        return vertices

    # def neck_angle(self, head, neck, trunk):
    #     zero_frame = [-90, -180, -180]
    #     REAR = self.point_poses['REAR']
    #     LEAR = self.point_poses['LEAR']
    #     HDTP = self.point_poses['HDTP']
    #     EAR = Point.mid_point(REAR, LEAR)
    #     RSHOULDER = self.point_poses['RSHOULDER']
    #     LSHOULDER = self.point_poses['LSHOULDER']
    #     C7 = self.point_poses['C7']
    #     # RPSIS = self.point_poses['RPSIS']
    #     # LPSIS = self.point_poses['LPSIS']
    #     PELVIS = self.point_poses['PELVIS']
    #
    #     HEAD_plane = Plane()
    #     HEAD_plane.set_by_pts(REAR, LEAR, HDTP)
    #     HEAD_coord = CoordinateSystem3D()
    #     HEAD_coord.set_by_plane(HEAD_plane, EAR, HDTP, sequence='yxz', axis_positive=True)
    #     NECK_angles = JointAngles()
    #     NECK_angles.ergo_name = {'flexion': 'flexion', 'abduction': 'L-bend', 'rotation': 'rotation'}  # lateral bend
    #     NECK_angles.set_zero(zero_frame, by_frame=False)
    #     NECK_angles.get_flex_abd(HEAD_coord, Point.vector(C7, PELVIS), plane_seq=['xy', 'yz'], flip_sign=[1, -1])
    #     NECK_angles.get_rot(LEAR, REAR, LSHOULDER, RSHOULDER)

    def back_angles(self, up_axis=[0, 1000, 0], zero_frame = [-90, 180, 180]):
        # todo: back correction
        C7 = self.point_poses['C7']
        # RPSIS = self.point_poses['RPSIS']
        # LPSIS = self.point_poses['LPSIS']
        RHIP = self.point_poses['RHIP']
        LHIP = self.point_poses['LHIP']
        RSHOULDER = self.point_poses['RSHOULDER']
        LSHOULDER = self.point_poses['LSHOULDER']
        # PELVIS_b = Point.mid_point(RPSIS, LPSIS)
        PELVIS = self.point_poses['PELVIS']

        BACK_plane = Plane()
        BACK_plane.set_by_vector(PELVIS, Point.create_const_vector(*up_axis, examplePt=PELVIS), direction=1)
        BACK_coord = CoordinateSystem3D()
        # BACK_RPSIS_PROJECT = BACK_plane.project_point(RPSIS)
        BACK_RHIP_PROJECT = BACK_plane.project_point(RHIP)
        BACK_coord.set_by_plane(BACK_plane, PELVIS, BACK_RHIP_PROJECT, sequence='zyx', axis_positive=True)
        BACK_angles = JointAngles()
        BACK_angles.ergo_name = {'flexion': 'flexion', 'abduction': 'L-flexion', 'rotation': 'rotation'}  #lateral flexion
        BACK_angles.set_zero(zero_frame)
        BACK_angles.get_flex_abd(BACK_coord, Point.vector(PELVIS, C7), plane_seq=['xy', 'yz'])
        # BACK_angles.get_rot(RSHOULDER, LSHOULDER, RPSIS, LPSIS, flip_sign=1)
        BACK_angles.get_rot(RSHOULDER, LSHOULDER, RHIP, LHIP, flip_sign=1)

        return BACK_angles

    def pose2REBA(self, motion_pred):
         # Step 1: compute angles
        pred_xyz = recover_from_ric((motion_pred).float(), self.nb_joints).reshape(1, -1, self.nb_joints, 3)

        version_2 = self.version_2
        if version_2:  # V2, add SMPL verts in loss
            pred_vert = self.get_smpl_mesh(pred_xyz)
            ## get SMPL vertID, to 66 keypoints
            marker_vids = self.marker_vids()['smpl']
            name_list = list(marker_vids.keys())
            idx_list = list(marker_vids.values())
            marker_vertices = pred_vert[:, idx_list, :]
            # marker_vids = marker_vids['smpl']
            # for i in range(len(name_list)):
            #     print(i, name_list[i])

        batch_size = pred_xyz.shape[1]
        ### trunk
        trunk_vec = pred_xyz[0, :, 12, :] - pred_xyz[0, :, 0, :]  # Root-->Neck_base
        y_pos_vec = torch.tensor([0, 1, 0]).repeat(batch_size, 1).float().cuda()  # same shape, with y +
        trunk_angle = self.anlge_3D(trunk_vec, y_pos_vec)  # trunk angle + - is the same score
        zero_frame = [-90, 180, 180]
        PELVIS = Point(pred_xyz[0, :, 0, :])
        NECK_BASE =  Point(pred_xyz[0, :, 12, :])
        RHIP = Point(pred_xyz[0, :, 2, :])
        LHIP = Point(pred_xyz[0, :, 1, :])
        LSHOULDER = Point(pred_xyz[0, :, 16, :])
        RSHOULDER = Point(pred_xyz[0, :, 17, :])
        BACK_plane = Plane()
        BACK_plane.set_by_vector(PELVIS, Point.create_const_vector(y_pos_vec, examplePt=PELVIS), direction=1)
        BACK_coord = CoordinateSystem3D()
        BACK_RHIP_PROJECT = BACK_plane.project_point(RHIP)
        BACK_coord.set_by_plane(BACK_plane, PELVIS, BACK_RHIP_PROJECT, sequence='zyx', axis_positive=True)
        BACK_angles = JointAngles()
        BACK_angles.ergo_name = {'flexion': 'flexion', 'abduction': 'L-flexion', 'rotation': 'rotation'}  # lateral flexion
        BACK_angles.set_zero(zero_frame)
        BACK_angles.get_flex_abd(BACK_coord, Point.vector(PELVIS, NECK_BASE), plane_seq=['xy', 'yz'])
        # BACK_angles.get_rot(RSHOULDER, LSHOULDER, RPSIS, LPSIS, flip_sign=1)
        BACK_angles.get_rot(RSHOULDER, LSHOULDER, RHIP, LHIP, flip_sign=1)


        ### upper_arm
        right_upper_arm_vec = pred_xyz[0, :, 19, :] - pred_xyz[0, :, 17, :]
        left_upper_arm_vec = pred_xyz[0, :, 18, :] - pred_xyz[0, :, 16, :]
        right_upper_arm_angle = np.pi - self.anlge_3D(right_upper_arm_vec, trunk_vec)
        left_upper_arm_angle = np.pi - self.anlge_3D(left_upper_arm_vec, trunk_vec)  # + - is the same score
        right_upper_arm_angle = np.pi - self.anlge_3D(right_upper_arm_vec, trunk_vec)
        left_upper_arm_angle = np.pi - self.anlge_3D(left_upper_arm_vec, trunk_vec)  # + - is the same score
        upper_arm_angle = torch.max(right_upper_arm_angle, left_upper_arm_angle)

        ### lower_arm
        right_lower_arm_vec = pred_xyz[0, :, 21, :] - pred_xyz[0, :, 19, :]
        left_lower_arm_vec = pred_xyz[0, :, 20, :] - pred_xyz[0, :, 18, :]
        right_lower_arm_angle = self.anlge_3D(right_lower_arm_vec, right_upper_arm_vec)
        left_lower_arm_angle = self.anlge_3D(left_lower_arm_vec, left_upper_arm_vec)
        lower_arm_angle = torch.max(right_lower_arm_angle, left_lower_arm_angle)


        if version_2:
            ### wrist: no wrist joint in the 22 kpts
            right_hand = (marker_vertices[0, :, 27, :] + marker_vertices[0, :, 28, :])/2
            left_hand = (marker_vertices[0, :, 29, :] + marker_vertices[0, :, 30, :]) / 2
            right_hand_vec = right_hand - pred_xyz[0, :, 21, :]
            left_hand_vec = left_hand - pred_xyz[0, :, 20, :]
            right_wrist_angle = self.anlge_3D(right_hand_vec, right_lower_arm_vec)
            left_wrist_angle = self.anlge_3D(left_hand_vec, left_lower_arm_vec)
            wrist_angle = torch.max(right_wrist_angle, left_wrist_angle)

            ### head
            HDTP = marker_vertices[0, :, 0, :]
            head_vec = HDTP - pred_xyz[0, :, 12, :]
            neck_angle = self.anlge_3D(head_vec, trunk_vec)

        ### leg
        right_upper_leg_vec = pred_xyz[0, :, 5, :] - pred_xyz[0, :, 2, :]
        left_upper_leg_vec = pred_xyz[0, :, 4, :] - pred_xyz[0, :, 1, :]
        right_lower_leg_vec = pred_xyz[0, :, 8, :] - pred_xyz[0, :, 5, :]
        left_lower_leg_vec = pred_xyz[0, :, 7, :] - pred_xyz[0, :, 4, :]

        right_knee_angle = self.anlge_3D(right_upper_leg_vec, right_lower_leg_vec)
        left_knee_angle = self.anlge_3D(left_upper_leg_vec, left_lower_leg_vec)
        knee_angle = torch.max(right_knee_angle, left_knee_angle)

        # Step 2: joint scores
        trunk = reba.get_trunk_score(BACK_angles.flexion, steepness=1)
        upper_arm = reba.get_upper_arm_score(BACK_angles.rotation, steepness=1)

        # upper_arm = reba.get_upper_arm_score(upper_arm_angle, steepness=1)
        lower_arm = reba.get_lower_arm_score(lower_arm_angle, steepness=1)
        leg = reba.get_leg_score(knee_angle, steepness=1)

        if version_2:
            wrist = reba.get_wrist_score(wrist_angle, steepness=1)
            neck = reba.get_neck_score(neck_angle, steepness=1)
        else:
            wrist = torch.ones_like(trunk)  # wrist = reba.get_wrist_score(angles, steepness=1)
            neck = torch.ones_like(trunk)  # neck = reba.get_neck_score(angles, steepness=1)


        # Step 3: REBA score
        score_a = reba.get_score_a(neck, leg, trunk)  # Score A value between 1 and 12
        score_b = reba.get_score_b(upper_arm, lower_arm, wrist)  # Score B value between 1 and 12
        score_c = reba.get_score_c(score_a, score_b)


        if self.verbose:
            print(f"Trunk angle: {trunk_angle}")
            print(f"Upper arm angle: {upper_arm_angle}")
            print(f"Lower arm angle: {lower_arm_angle}")
            print(f"Knee angle: {knee_angle}")
            print(f"Trunk score: {trunk}")
            print(f"Upper arm score: {upper_arm}")
            print(f"Lower arm score: {lower_arm}")
            print(f"Leg score: {leg}")
            print(f"Wrist score: {wrist}")
            print(f"Neck score: {neck}")
            print(f"REBA score A: {score_a}")
            print(f"REBA score B: {score_b}")
            print(f"REBA score C: {score_c}")
            print(f"Average REBA score C: {torch.mean(score_c)}")
        return score_c

        # # Step 4: REBA action level
        # action_level = reba.get_action_level(score_c)

        # Step 5: Loss
        # loss = (action_level)**2/16  # quadratic loss normalized between 0 and 1



    def forward(self, motion_pred, motion_gt):
        score_c = self.pose2REBA(motion_pred)
        score_c_gt = self.pose2REBA(motion_gt)
        score_diff = score_c - score_c_gt*self.REBA_improve


        self.ave_REBA_score = torch.mean(score_c)
        self.max_REBA_score = torch.max(score_c)

        # max
        # loss = (self.max_REBA_score -1)** 2 /  15**2  # quadratic loss normalized between 0 and 1

        # sum
        if False:
            loss = (score_c-1) ** 2 / 15**2  # quadratic loss normalized between 0 and 1
            loss = torch.sum(loss)
        elif True:
            # leaky relu
            loss = torch.nn.functional.leaky_relu(score_diff, negative_slope=0.1)
            loss = torch.sum(loss)
        elif False:
            # leaky relu
            loss = torch.nn.functional.leaky_relu(score_diff, negative_slope=0.1)** 2 
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
