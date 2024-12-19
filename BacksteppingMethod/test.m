% Description: None
% Author: Bin Peng
% Email: pb20020816@163.com
% Date: 2024-12-15 15:54:45
% LastEditTime: 2024-12-15 19:42:45
clear
syms u v w freq amp bias u_dot v_dot w_dot k_1 k_r r theta

% 定义 a2 和 a1
a2 = -0.1103 * u + 13.9448 * v + 12.8666 * w;
a1 = -3.1762 * u + 0.0279 * v - 2.4948 * w;

% 定义矩阵 C, D, 和 M
C = [  0,           0,         -17*v-a2;
       0,           0,         17*u+a1;
       17*v+a2, -17*u-a1,         0];

D = -[  0.5996,  2.4967, -1.9539;
      -0.3623, 11.0042,  9.2471;
       0.2295, -6.5004, -7.3168];

M = [  20.6222,  -0.0279,   2.4647;
        0.1103,   3.5012, -12.8056;
       -0.1222,   8.2168,   9.2795];

% 定义速度和推力向量
Velocity_dot = [u_dot; v_dot; w_dot];
Velocity = [u; v; w];
% freq = 1;
% amp = 30;
thrust = (0.0514 * exp(0.03102 * (freq - 0.4961)^2) - ...
          1.7630 * exp(-2.2080 * (freq - 1.8670)^2) - ...
          0.8956 * exp(-1.9130 * (freq - 1.0820)^2)) * ...
         (0.0085 * amp^2 - 0.6171 * amp + 4.0280);
F_thrust = [ thrust * (1-(bias/180*pi)^2/2);
             thrust * (bias/180*pi);
             thrust * 0.2 * (bias/180*pi)];

syms thrust_part1 thrust_part2
% thrust_part1 = (0.0514 * exp(0.03102 * (freq - 0.4961)^2) - ...
%           1.7630 * exp(-2.2080 * (freq - 1.8670)^2) - ...
%           0.8956 * exp(-1.9130 * (freq - 1.0820)^2))
% thrust_part2 = (0.0085 * amp^2 - 0.6171 * amp + 4.0280);

F_thrust = [ thrust_part1 * thrust_part2 * (1-bias^2/2);
             thrust_part1 * thrust_part2 * bias;
             -thrust_part1 * thrust_part2 * 0.1 * bias];

% 构建方程组 eq1
eq1 = M * Velocity_dot == -(C + D) * Velocity + F_thrust;
solution1 = solve(eq1, [u_dot, v_dot, w_dot]);

% 显示加速度的解析解（浮点形式）
disp('加速度的解析解 (浮点形式):');
solution1_float = structfun(@vpa, solution1, 'UniformOutput', false);
disp(solution1_float);

% 计算 X 和 Y
du_dot = solution1_float.u_dot;
dv_dot = solution1_float.v_dot;
dw_dot = solution1_float.w_dot;

X = dv_dot * sin(theta) - du_dot * cos(theta) + ...
    (k_1 + k_r) * (-u * cos(theta) + v * sin(theta)) + ...
    k_1 * k_r * r + ...
    (u * sin(theta) + v * cos(theta)) * (u / r * sin(theta) + v / r * cos(theta) + w);

Y = 0;
eq2 = X == Y;
solution2 = solve(eq2, thrust_part2);

% 显示解 thrust_part2 (浮点形式)
disp('thrust_part2 的解析解 (浮点形式):');
solution2_float = vpa(solution2);
disp(solution2_float);

% u=0;v=0;w=0;r=5;theta=pi/4;amp=pi;
% thrust_part1 = (0.0514 * exp(0.03102 * (1 - 0.4961)^2) - ...
%           1.7630 * exp(-2.2080 * (1 - 1.8670)^2) - ...
%           0.8956 * exp(-1.9130 * (1 - 1.0820)^2))
% thrust_part2 = (sin(theta)*(0.024309390060648402687884086858897*u - 0.74229599144516638561745720300377*v + 0.68030057029160631256205361775987*w - 0.067412500696260895256598449361081*w*(13.8238*u + 0.0279*v - 2.4948*w + 9.2471) + 0.09297788244746423471491001773451*v*(13.8238*u + 0.0279*v - 2.4948*w + 6.5004) - 0.09297788244746423471491001773451*u*(30.9448*v - 0.1103*u + 12.8666*w + 0.2295) + 0.00019039183056524293459228042798975*w*(30.9448*v - 0.1103*u + 12.8666*w + 1.9539)) - 1.0*(v*cos(theta) + u*sin(theta))*(w + (v*cos(theta))/r + (u*sin(theta))/r) + cos(theta)*(0.026428325163817343131666137978312*u + 0.2003144416991875090936946356628*v + 0.021288079293693457416525574199766*w + 0.0072140877996593715018339147058815*w*(13.8238*u + 0.0279*v - 2.4948*w + 9.2471) + 0.0029094794573711810376839020063096*v*(13.8238*u + 0.0279*v - 2.4948*w + 6.5004) - 0.0029094794573711810376839020063096*u*(30.9448*v - 0.1103*u + 12.8666*w + 0.2295) - 0.048435605693185345941928894723571*w*(30.9448*v - 0.1103*u + 12.8666*w + 1.9539)) + (u*cos(theta) - 1.0*v*sin(theta))*(k_1 + k_r) - 1.0*k_1*k_r*r)/(sin(theta)*(0.00019039183056524293459228042798975*thrust_part1*cos(bias) + 0.086008077185753742199580452907983*thrust_part1*sin(bias)) - 1.0*cos(theta)*(0.048435605693185345941928894723571*thrust_part1*cos(bias) + 0.0066321919081851352942971343046195*thrust_part1*sin(bias)))
% 
% eq3 = thrust_part2 == (0.0085 * amp^2 - 0.6171 * amp + 4.0280)
% vpa(eq3,5)
% 
% solution3 = solve(eq3, bias);
% solution3_float = vpa(solution3);
% disp(solution3_float);