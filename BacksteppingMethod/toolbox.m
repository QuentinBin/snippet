% 自定义模型
model = @(params, x) params(1)*sin(params(2)*x) + params(3);

% 输入输出数据
x_data = [input data];
y_data = [output data];

% 定义目标函数
error_func = @(params) sum((model(params, x_data) - y_data).^2);

% 使用fminunc进行优化
initial_params = [1, 1, 0];  % 初始猜测的参数
options = optimoptions('fminunc', 'Algorithm', 'quasi-newton');
optimized_params = fminunc(error_func, initial_params, options);
