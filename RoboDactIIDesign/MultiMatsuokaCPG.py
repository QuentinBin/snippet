'''
Description: Multi-Module Matsuoka CPG with Phase Coupling
Author: Bin Peng
Email: pb20020816@163.com
Date: 2025-07-07 14:03:30
LastEditTime: 2025-07-08 10:14:18
'''
import numpy as np
import matplotlib.pyplot as plt

class MultiMatsuokaCPG:
    def __init__(self, num_modules=3, tau_r=0.1, tau_a=0.2, alpha=2.0, beta=2.0):
        """
        初始化多模块Matsuoka CPG网络
        
        参数:
        num_modules: CPG模块数量
        tau_r: 恢复时间常数
        tau_a: 适应时间常数
        alpha: 抑制连接强度
        beta: 适应连接强度
        """
        self.num_modules = num_modules
        self.tau_r = tau_r
        self.tau_a = tau_a
        self.alpha = alpha
        self.beta = beta
        
        # 耦合参数
        self.coupling_strength = 0.5  # 模块间耦合强度
        self.phase_delays = np.zeros(num_modules)  # 相位差
        
        # 延迟缓冲区用于实现真正的相位差
        self.delay_buffer_size = 1000  # 缓冲区大小
        self.delay_buffers = {
            'y1': np.zeros((num_modules, self.delay_buffer_size)),
            'y2': np.zeros((num_modules, self.delay_buffer_size))
        }
        self.buffer_index = 0
        
        # 初始化状态变量
        self.x1 = np.zeros(num_modules)
        self.x2 = np.zeros(num_modules)
        self.v1 = np.zeros(num_modules)
        self.v2 = np.zeros(num_modules)
        self.y1 = np.zeros(num_modules)
        self.y2 = np.zeros(num_modules)
        
        # 历史记录
        self.history = {
            'x1': [], 'x2': [], 'v1': [], 'v2': [], 'y1': [], 'y2': []
        }
        
    def set_phase_delays(self, delays):
        """设置各模块的相位差（弧度）"""
        self.phase_delays = np.array(delays)
        
    def set_coupling_strength(self, strength):
        """设置模块间耦合强度"""
        self.coupling_strength = strength
        
    def set_initial_conditions(self, x1_init=None, x2_init=None, v1_init=None, v2_init=None):
        """设置初始条件"""
        if x1_init is not None:
            self.x1 = np.array(x1_init)
        else:
            self.x1 = np.random.uniform(-0.1, 0.1, self.num_modules)
            
        if x2_init is not None:
            self.x2 = np.array(x2_init)
        else:
            self.x2 = -self.x1
            
        if v1_init is not None:
            self.v1 = np.array(v1_init)
        else:
            self.v1 = np.random.uniform(-0.1, 0.1, self.num_modules)
            
        if v2_init is not None:
            self.v2 = np.array(v2_init)
        else:
            self.v2 = -self.v1
    
    def get_delayed_output(self, module_idx, delay_steps):
        """获取延迟的输出"""
        if delay_steps <= 0:
            return self.y1[module_idx] - self.y2[module_idx]
        
        # 计算延迟后的索引
        delayed_index = (self.buffer_index - delay_steps) % self.delay_buffer_size
        y1_delayed = self.delay_buffers['y1'][module_idx, delayed_index]
        y2_delayed = self.delay_buffers['y2'][module_idx, delayed_index]
        
        return y1_delayed - y2_delayed
    
    def compute_coupling_input_with_delay(self, module_idx, dt):
        """使用延迟缓冲区计算耦合输入 - 真正的相位差"""
        coupling_input = 0.0
        
        for j in range(self.num_modules):
            if j != module_idx:
                # 相邻模块的耦合
                if abs(j - module_idx) == 1 or (module_idx == 0 and j == self.num_modules - 1) or (module_idx == self.num_modules - 1 and j == 0):
                    # 将相位差转换为时间延迟步数
                    phase_diff = self.phase_delays[j] - self.phase_delays[module_idx]
                    # 假设一个完整周期对应2π，估算周期时间
                    estimated_period = 1.0  # 秒
                    delay_time = (phase_diff / (2 * np.pi)) * estimated_period
                    delay_steps = int(abs(delay_time) / dt)
                    
                    # 获取延迟的输出
                    delayed_output = self.get_delayed_output(j, delay_steps)
                    coupling_input += self.coupling_strength * delayed_output
                    
        return coupling_input
    
    def compute_coupling_input(self, module_idx, t):
        """计算来自其他模块的耦合输入 - 真正的相位差实现"""
        coupling_input = 0.0
        
        for j in range(self.num_modules):
            if j != module_idx:
                # 相邻模块的耦合（也可以设计为全连接）
                if abs(j - module_idx) == 1 or (module_idx == 0 and j == self.num_modules - 1) or (module_idx == self.num_modules - 1 and j == 0):
                    # 方法1：使用时间延迟的正弦波作为耦合输入（真正的相位差）
                    phase_diff = self.phase_delays[j] - self.phase_delays[module_idx]
                    # 创建一个具有相位差的振荡输入
                    phase_coupling = np.sin(2 * np.pi * 1.0 * t + phase_diff)  # 1Hz的参考频率
                    coupling_input += self.coupling_strength * (self.y1[j] - self.y2[j]) * (1 + 0.5 * phase_coupling)
                    
        return coupling_input
    
    def step(self, dt, u_ext=1.0, t=0.0):
        """单步更新"""
        # 保存当前状态
        x1_new = self.x1.copy()
        x2_new = self.x2.copy()
        v1_new = self.v1.copy()
        v2_new = self.v2.copy()
        
        # 更新每个模块
        for i in range(self.num_modules):
            # 选择耦合方法：
            # 方法1：时间相关的相位耦合（当前使用）
            # coupling_input = self.compute_coupling_input(i, t)
            
            # 方法2：延迟缓冲区（真正的相位差）- 取消注释以使用
            coupling_input = self.compute_coupling_input_with_delay(i, dt)
            
            # Matsuoka神经元方程
            dx1 = (-self.x1[i] - self.beta * self.v1[i] - self.alpha * self.y2[i] + u_ext + coupling_input) / self.tau_r
            dx2 = (-self.x2[i] - self.beta * self.v2[i] - self.alpha * self.y1[i] + u_ext + coupling_input) / self.tau_r
            
            x1_new[i] = self.x1[i] + dx1 * dt
            x2_new[i] = self.x2[i] + dx2 * dt
            
            # ReLU激活
            self.y1[i] = max(0.0, x1_new[i])
            self.y2[i] = max(0.0, x2_new[i])
            
            # 适应变量更新
            dv1 = (-self.v1[i] + self.y1[i]) / self.tau_a
            dv2 = (-self.v2[i] + self.y2[i]) / self.tau_a
            
            v1_new[i] = self.v1[i] + dv1 * dt
            v2_new[i] = self.v2[i] + dv2 * dt
            
            # 更新延迟缓冲区
            self.delay_buffers['y1'][i, self.buffer_index] = self.y1[i]
            self.delay_buffers['y2'][i, self.buffer_index] = self.y2[i]
        
        # 更新状态
        self.x1 = x1_new
        self.x2 = x2_new
        self.v1 = v1_new
        self.v2 = v2_new
        self.buffer_index = (self.buffer_index + 1) % self.delay_buffer_size
        
        # 记录历史
        self.history['x1'].append(self.x1.copy())
        self.history['x2'].append(self.x2.copy())
        self.history['v1'].append(self.v1.copy())
        self.history['v2'].append(self.v2.copy())
        self.history['y1'].append(self.y1.copy())
        self.history['y2'].append(self.y2.copy())

# 仿真参数
T_total = 10.0
dt = 0.001
steps = int(T_total / dt)

# 创建多模块CPG网络
num_modules = 4  # 4个CPG模块
cpg = MultiMatsuokaCPG(num_modules=num_modules, tau_r=0.1, tau_a=0.2, alpha=2.0, beta=2.0)

# 设置相位差（弧度）- 几种不同的选择：

# 选项1：严格的π/4间隔 (45°间隔)
phase_delays_option1 = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# 选项2：严格的π/2间隔 (90°间隔) 
phase_delays_option2 = [0, np.pi/2, np.pi, 3*np.pi/2]

# 选项3：等间隔的2π/4 = π/2间隔
phase_delays_option3 = [0, np.pi/2, np.pi, 3*np.pi/2]

# 选项4：鱼类游泳的典型相位差模式 (游泳波形)
phase_delays_option4 = [0, np.pi/6, np.pi/3, np.pi/2]  # 30°, 60°, 90°

# 选项5：严格的π/3间隔 (60°间隔)
phase_delays_option5 = [0, np.pi/3, 2*np.pi/3, np.pi]

# 选择使用哪种相位差模式
phase_delays = phase_delays_option2  # 可以改成option2, option3等

cpg.set_phase_delays(phase_delays)

# 设置耦合强度
cpg.set_coupling_strength(0.3)

# 设置初始条件
cpg.set_initial_conditions(
    x1_init=[0.1, 0.1, 0.1, 0.1],
    x2_init=[-0.1, -0.1, -0.1, -0.1],
    v1_init=[0.1, 0.1, 0.1, 0.1],
    v2_init=[-0.1, -0.1, -0.1, -0.1]
)

# 仿真主循环
for i in range(steps):
    t_curr = i * dt
    
    u_ext = 1.0
    
    cpg.step(dt, u_ext, t_curr)

# 转换历史数据为numpy数组
t = np.linspace(0, T_total, steps)
y1_history = np.array(cpg.history['y1'])
y2_history = np.array(cpg.history['y2'])

# 计算输出差值（控制信号）
output_signals = y1_history - y2_history

# 可视化
plt.figure(figsize=(15, 10))

# subplot 1: 所有模块的输出
plt.subplot(3, 1, 1)
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
for i in range(num_modules):
    plt.plot(t, y1_history[:, i], '--', color=colors[i], alpha=0.6, label=f'Module {i+1} - Neuron 1')
    plt.plot(t, y2_history[:, i], ':', color=colors[i], alpha=0.6, label=f'Module {i+1} - Neuron 2')
plt.xlabel("Time (s)")
plt.ylabel("Neural Output")
plt.title("Multi-Module Matsuoka CPG - Neural Outputs")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# subplot 2: 各模块的控制信号
plt.subplot(3, 1, 2)
for i in range(num_modules):
    plt.plot(t, output_signals[:, i], color=colors[i], linewidth=2, label=f'Module {i+1} Output')
plt.axvline(x=2.7, color='red', linestyle='--', alpha=0.7, label='Input Change')
plt.xlabel("Time (s)")
plt.ylabel("Control Signal")
plt.title("Multi-Module CPG Control Signals (with Phase Coupling)")
plt.legend()
plt.grid(True)

# subplot 3: 相位关系展示
plt.subplot(3, 1, 3)
# 选择一个时间段来展示相位关系
start_idx = int(5.0 / dt)  # 从5秒开始
end_idx = int(7.0 / dt)    # 到7秒结束
t_phase = t[start_idx:end_idx]
for i in range(num_modules):
    plt.plot(t_phase, output_signals[start_idx:end_idx, i], 
             color=colors[i], linewidth=2, label=f'Module {i+1} (Phase: {phase_delays[i]:.2f} rad)')
plt.xlabel("Time (s)")
plt.ylabel("Control Signal")
plt.title("Phase Relationship Detail (5-7 seconds)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 打印相位差信息
print("CPG模块配置:")
print(f"模块数量: {num_modules}")
print(f"耦合强度: {cpg.coupling_strength}")
print("相位差设置:")
for i, phase in enumerate(phase_delays):
    print(f"  模块 {i+1}: {phase:.3f} 弧度 ({np.degrees(phase):.1f}°)")

print("\n可选的相位差模式:")
print("选项1 (π/4间隔): [0°, 45°, 90°, 135°]")
print("选项2 (π/2间隔): [0°, 90°, 180°, 270°]") 
print("选项3 (等间隔):   [0°, 90°, 180°, 270°]")
print("选项4 (鱼类游泳): [0°, 30°, 60°, 90°]")
print("选项5 (π/3间隔): [0°, 60°, 120°, 180°]")
print(f"当前使用: {[f'{np.degrees(p):.1f}°' for p in phase_delays]}")

print("\n⚠️ 重要说明：")
print("1. 当前的相位差实现方法：")
print("   - 方法1（正在使用）: 使用时间相关的正弦波调制耦合强度")
print("   - 方法2（可选）: 使用延迟缓冲区实现真正的时间延迟")
print("2. 设置的phase_delays参数并不直接等于最终的相位差")
print("3. 实际相位差取决于CPG的内在动力学和耦合强度")
print("4. 要使用真正的延迟相位差，请在step方法中启用方法2")
print("5. 观察图形中的相位关系来验证实际的相位差效果")
