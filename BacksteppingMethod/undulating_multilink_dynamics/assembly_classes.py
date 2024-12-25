import numpy as np

class Assembly():
    def __init__(self):
        self.links = []  # list of all objects
        self.joints = []   


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
            "position": np.array(joint_position, dtype=float)
            "angular_velocity": 0.0 # unit:rad/s
        })
    #---------------------------------------------------#