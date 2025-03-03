'''
Description: None
Author: Bin Peng
Email: pb20020816@163.com
Date: 2025-03-03 09:04:58
LastEditTime: 2025-03-03 09:05:02
'''
import numpy as np

def added_mass_revolution_ellipsoid(a, c, rho=1000):
    """
    计算旋转对称椭球的附加质量
    """
    if c > a:
        e = np.sqrt(1 - (a**2 / c**2))  # 细长椭球
    else:
        e = np.sqrt(1 - (c**2 / a**2))  # 扁平椭球
    
    if e == 0:
        volume = (4/3) * np.pi * a**2 * c
        added_mass = 0.5 * rho * volume
        return added_mass, added_mass, added_mass

    e = max(e, 1e-6)  # 避免计算奇点

    # 计算 ln((1+e)/(1-e)) 时避免溢出
    ln_term = np.log((1 + e) / (1 - e)) if e < 0.999 else (2 * e + (2/3) * e**3)

    beta_z = (1 - e**2) / e**2 - ln_term / (2 * e**3)
    beta_x = 1 / (2 * (1 - e**2)) + ln_term / (4 * e**3)

    # 确保 beta 正值
    beta_z = max(beta_z, 1e-6)
    beta_x = max(beta_x, 1e-6)

    m_az = rho * (4/3) * np.pi * a**2 * c / beta_z
    m_ax = m_ay = rho * (4/3) * np.pi * a * c**2 / beta_x

    return m_ax, m_ay, m_az

# 示例
a, c = 0.1, 0.2
added_mass = added_mass_revolution_ellipsoid(a, c)
print(f"Added Mass: mx = {added_mass[0]:.3f}, my = {added_mass[1]:.3f}, mz = {added_mass[2]:.3f}")
