'''
Description: Geometric Methods for Modeling and Control of Free-Swimming Fin-Actuated Underwater Vehicles
Author: Bin Peng
Email: pb20020816@163.com
Date: 2024-11-20 09:10:33
LastEditTime: 2024-11-20 09:26:53
'''
import numpy as np

# Body Configration. Unit: Kg,m
Mass_body = 3.0
Length_body = 0.40
Width_body = 0.06
Height_body = 0.09
C_lb_12 = 0.12 # deg^-1
C_lb_13 = 0.03
C_db_0 = 0.0
C_db_12 = 0.02
C_db_13 = 0.01
C_m = 2.0 # kgxm^2
Area_12 = 0.04 # m^2
Area_13 = 0.032

# Penduncle
Mass_penduncle = 0.1
Length_penduncle = 0.085

# Tail
Mass_tail = 0.15
Chord_tail = 0.08 # m
Span_tail = 0.25

# Pectoral Fin
Chord_fin = 0.10
Span_fin = 0.35