'''
Description: Geometric Methods for Modeling and Control of Free-Swimming Fin-Actuated Underwater Vehicles
Author: Bin Peng
Email: pb20020816@163.com
Date: 2024-11-20 09:10:33
LastEditTime: 2024-11-20 14:37:29
'''
import numpy as np



class Fish():
    def __init__(self):
        # Body Configration. Unit: Kg,m
        self._Mass_body = 3.0
        self._Length_body = 0.40
        self._Width_body = 0.06
        self._Height_body = 0.09
        self._C_lb_12 = 0.12 # deg^-1
        self._C_lb_13 = 0.03
        self._C_db_0 = 0.0
        self._C_db_12 = 0.02
        self._C_db_13 = 0.01
        self._C_m = 2.0 # kgxm^2
        self._Area_12 = 0.04 # m^2
        self._Area_13 = 0.032

        # Peduncle
        self._Mass_peduncle = 0.1
        self._Length_peduncle = 0.085

        # Tail
        self._Mass_tail = 0.15
        self._Chord_tail = 0.08 # m
        self._Span_tail = 0.25

        # Pectoral Fin
        self._Chord_fin = 0.10
        self._Span_fin = 0.35

        # 
        self._MassMatrix_Body, self._InertiaMatrix_Body = \
            self._CalculateMassInertiaMatix_Ellipsoid(self._Length_body, self._Width_body, self._Height_body, 1, self._Mass_body)
        self._MassMatrix_Tail, self._InertiaMatrix_Tail = \
            self._CalculateMassInertiaMatix_FlatPlate(self._Chord_tail, self._Span_tail, 1, self._Mass_tail)
    
    def _CalculateMassInertiaMatix_Ellipsoid(self, a1, a2, a3, rho, m):
        """
        计算椭球体的附加质量和附加惯性
        :param a1: 椭球体的长半轴 (x1)
        :param a2: 椭球体的短半轴 (x2)
        :param a3: 椭球体的短半轴 (x3)
        :param rho: 流体密度
        :param m: 椭球体质量
        :return: mass+added mass matrxi; inertia
        """
        # 附加质量
        added_mass = (4 / 3) * np.pi * rho * a1 * a2 * a3

        # 附加惯性
        term1 = (1 / 5) * m  # 椭球体自身惯性系数
        term2 = (4 / 15) * np.pi * rho * a1 * a2 * a3  # 流体附加惯性系数

        # 绕 x1, x2, x3 的惯性
        Jx1 = term1 * (a2**2 + a3**2) + term2 * (a2**2 + a3**2)
        Jx2 = term1 * (a1**2 + a3**2) + term2 * (a1**2 + a3**2)
        Jx3 = term1 * (a1**2 + a2**2) + term2 * (a1**2 + a2**2)

        mass_matrix = (m+added_mass)*np.eye(3)
        inertia_matrix = np.diag([Jx1,Jx2,Jx3])

        # print('mass_matrix:', mass_matrix)
        # print('inertial_matrix:', inertia_matrix)
        return mass_matrix, inertia_matrix
    
    def _CalculateMassInertiaMatix_FlatPlate(self, l, h, rho, m):
        """
        计算平板的附加质量和附加惯性
        :param l: 平板的长度
        :param h: 平板的高度
        :param rho: 流体密度
        :param m:mass
        :return: 附加质量 (added_mass_x2), 附加惯性 (Jx1, Jx2, Jx3)
        """
        # 附加质量（x2方向）
        added_mass_x2 = (1 / 4) * np.pi * rho * l**2 * h
        mass_matrix = np.diag([m, added_mass_x2+m, m])
        # 附加惯性（绕x1和x3方向）
        J_added_x3 = 2 * np.pi * rho * (l / 4)**4 * h
        J_added_x1 = 2 * np.pi * rho * (h / 4)**4 * l
        # 自身惯性矩阵
        inertia_matrix = np.diag([
            (1 / 12) * m * (h**2) + J_added_x1,  # I_x1
            (1 / 12) * m * (l**2),  # I_x2
            (1 / 12) * m * (l**2 + h**2) + J_added_x3   # I_x3
        ])

        # 绕x2方向的惯性为零
        return mass_matrix, inertia_matrix
    
    def _CalculateLiftForces(d, Vf1, Vf2, omega_f):
        """
        Calculate the lift forces for a heaving and pitching hydrofoil.
        
        Parameters:
        d (float): Quarter-chord location of the foil
        Vf1 (float): Speed of the foil in the body-fixed x1 direction
        Vf2 (float): Speed of the foil in the body-fixed x2 direction
        omega_f (float): Angular velocity of the foil about its leading edge
        
        Returns:
        tuple: (Lf1, Lf2), forces along the chord and out of the chord/span plane
        """
        # Calculate the forces
        Lf1 = 4 * np.pi * d * Vf1 * (Vf1 + d * omega_f)
        Lf2 = 4 * np.pi * d * Vf2 * (Vf1 + d * omega_f)
        
        return Lf1, Lf2
    

if __name__ == '__main__':
    fish = Fish()