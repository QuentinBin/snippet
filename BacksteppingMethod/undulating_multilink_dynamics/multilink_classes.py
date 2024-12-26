'''
Description: 多连杆鱼的动力学建模
Author: Bin Peng
Email: pb20020816@163.com
Date: 2024-12-25 14:08:19
LastEditTime: 2024-12-25 18:13:22
'''
import numpy as np

class BodyLinks():
    def __init__(self):
        # kinematic:
        self._mass_center = np.zeros(3) # mass center (in parental joint's coordinate system)
        self._global_SE3 = np.eye(4) # Global SE3
        self._global_se3 = np.zeros(6) # Global se3

        # hydrodynamic:
        self._CL_hor = 0.0 # horizontal lift coeffcient
        self._CD_hor = 0.0 # horizontal drag coefficent
        self._cross_area = 0.0 # cross area


    #--------------- EXTERNAL FUNCTIONS -----------------#
    def set_mass_center(self, mass_center):
        self._mass_center = mass_center
        return mass_center
    
    def set_global_SE3(self, global_SE3:np.array):
        self._global_SE3 = global_SE3
        return global_SE3
    
    def set_global_se3(self, global_se3:np.array):
        self._global_se3 = global_se3
        return global_se3
    
    #---------------------------------------------------#