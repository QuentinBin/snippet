clear
syms u v w freq_l  freq_r  u_dot v_dot w_dot k_1 k_r r theta delta_amp
l=0.2;
% 定义矩阵 C, D, 和 M
a2 = -0.1103 * u + 1.9448 * v + 1.8666 * w;
a1 = -3.1762 * u + 0.0279 * v - 2.4948 * w;
C = [  0,           0,         -17.446*v-a2;
       0,           0,         17.446*u+a1;
       17*v+a2, -17.446*u-a1,         0];

D = -[  0.5996,  2.4967, -1.9539;
      -0.3623, 11.0042,  9.2471;
       0.2295, -6.5004, -7.3168];

M = [  20.6222,  -0.0279,   2.4647;
        0.1103,   23.5012, -1.28056;
       -0.1222,   .82168,   9.2795];

% 定义速度和推力向量
Velocity_dot = [u_dot; v_dot; w_dot];
Velocity = [u; v; w];

amp_l = 30 + delta_amp;
amp_r = 30 - delta_amp;
F_pec_l = 0.5*((0.9262*freq_l^3 -1.6480*freq_l^2 + 1.5960*freq_l + 0.6419)*(0.0442*amp_l - 1.5290) +  (0.1016*freq_l^3 +1.5580*freq_l^2 -2.0650*freq_l + 1.4700)*(-0.1958*amp_l +3.0170) +  (0.0535*freq_l^3 +0.7496*freq_l^2 +1.3190*freq_l + 0.3941)*(0.1191*amp_l -0.7917));
F_pec_r = 0.5*((0.9262*freq_r^3 -1.6480*freq_r^2 + 1.5960*freq_r + 0.6419)*(0.0442*amp_r - 1.5290) +  (0.1016*freq_r^3 +1.5580*freq_r^2 -2.0650*freq_r + 1.4700)*(-0.1958*amp_r +3.0170) +  (0.0535*freq_r^3 +0.7496*freq_r^2 +1.3190*freq_r + 0.3941)*(0.1191*amp_r -0.7917));
   
F_pec_trust = [F_pec_l+F_pec_r; 0; F_pec_r*l-F_pec_l*l];

% 构建方程组 eq1
eq1 = M * Velocity_dot == -(C + D) * Velocity + F_pec_trust;
solution1 = solve(eq1, [u_dot, v_dot, w_dot]);
% 显示加速度的解析解（浮点形式）
disp('加速度的解析解 (浮点形式):');
solution1_float = structfun(@vpa, solution1, 'UniformOutput', false);
disp(solution1_float);

% 计算 X 和 Y
du_dot = solution1_float.u_dot;
dv_dot = solution1_float.v_dot;
dw_dot = solution1_float.w_dot;
% 构建方程组 eq2
eq2 = dv_dot * sin(theta) - du_dot * cos(theta) + ...
    (k_1 + k_r) * (-u * cos(theta) + v * sin(theta)) + ...
    k_1 * k_r * r + ...
    (u * sin(theta) + v * cos(theta)) * (u / r * sin(theta) + v / r * cos(theta) + w) ...
    ==0;
solution2 = solve(eq2, [ delta_amp ]);
% 显示控制量的解析解（浮点形式）
disp('加速度的解析解 (浮点形式):');
disp(solution2)