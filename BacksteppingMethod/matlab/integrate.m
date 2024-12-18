
% 定义常量参数
l_y = 0.15;
l_x = 0.3;
freq = 1;
n = 0.75;
amp = 0.15;

% 定义积分的函数 f_x, f_y, f_t 和 dL_x
f_x = @(x, y, t) (amp * 2 * n * pi) / (l_x * l_y) * y .* sin(-2*pi*freq*t + (x - l_x) / l_x * 2 * n * pi);
f_y = @(x, y, t) (amp) / (l_y) * sin(-2*pi*freq*t + (x - l_x) / l_x * 2 * n * pi);
f_t = @(x, y, t) (-amp * 2 * pi * freq) / (l_y) * y .* cos(-2*pi*freq*t + (x - l_x) / l_x * 2 * n * pi);

% 定义 dL_x
dL_x = @(x, y, t) (f_t(x, y, t).^2 .* sqrt(f_x(x, y, t).^2 + f_y(x, y, t).^2 + 1) ...
    .* (-f_x(x, y, t)) ./ sqrt(f_x(x, y, t).^2 + f_y(x, y, t).^2 ));

% 使用 integral3 进行数值积分
L_x = integral3(@(x, y, t) dL_x(x, y, t), 0, l_x, 0, l_y, 0, 1, 'RelTol', 1e-6, 'AbsTol', 1e-6);

% 输出结果
disp(L_x)
