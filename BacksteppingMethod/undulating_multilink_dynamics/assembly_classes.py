'''
Description: 多连杆鱼的装配体类
Author: Bin Peng
Email: pb20020816@163.com
Date: 2024-12-25 14:49:12
LastEditTime: 2024-12-25 23:40:03
'''
import numpy as np
from multilink_classes import BodyLinks
class Assembly():
    def __init__(self):
        self.links = []  # list of all objects
        self.joints: = []  

    #--------------- INTERNAL FUNCTIONS -----------------#
    def _compute
    #----------------------------------------------------#

    #--------------- EXTERNAL FUNCTIONS -----------------#
    def add_links(self, link):
        self.links.append(link)

    def add_joint(self, link1_idx, link2_idx, joint_position=[0,0,0], joint_axis=[0,0,1]):
        """
        :param link1_idx: index of parent link
        :param link2_idx: index of child link
        :param joint_axis: rotation axis in parent coordinate system
        :param joint_position: joint positon in parent coordinate system
        """
        self.joints.append({
            "parent": link1_idx,
            "child": link2_idx,
            "axis": np.array(joint_axis, dtype=float),
            "position": np.array(joint_position, dtype=float),
            "angle":0.0,
            "angular_velocity": 0.0, # unit:rad/s
        })

    def set_joint_rotate(self, joint_id, theta, omega):
        self.joints[joint_id]['angle'] = theta
        self.joints[joint_id]['angular_velocity'] = omega
    
    def update_baselinks(self, dt):

    def update_sublinks(self):
        for joint in self.joints:
            # Assumed that parent link is already finished
            parent = self.links[joint["parent"]]
            child = self.links[joint["child"]]
            axis_local = joint["axis"]
            position_local = joint["position"]

            # Transe joint to global CS
            axis_global = parent._global_SE3[:3,:3].dot(axis_local)
            position_global = parent._global_SE3[:3,:3].dot(position_local) + parent._global_SE3[:3,3]

            # Update Child link's orientation
            R_child_in_parent = np.array([
                [np.cos(joint['angle']), -np.sin(joint['angle']), 0],
                [np.sin(joint['angle']), np.cos(joint['angle']), 0],
                [0,0,1]
            ])
            child._global_SE3[:3,:3] = parent._global_SE3[:3,:3].dot(R_child_in_parent)
            child._global_SE3[:3,3] = position_global
            child._global_se3[:3] = parent._global_se3[:3] + joint['angular_velocity'] * axis_global
            child._global_se3[3:6] = parent._global_se3[3:6] + np.cross(parent._global_se3[:3], position_global-parent._global_SE3[:3,3])
    #---------------------------------------------------#