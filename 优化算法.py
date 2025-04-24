import pickle
import sys
from cProfile import label
from deap.benchmarks.tools import scipy_imported

# 添加EnergyPlus路径

sys.path.append('F:/energyplus23.2')
from pyenergyplus.api import EnergyPlusAPI

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import pandas as pd
from sklearn.neural_network import MLPRegressor
import seaborn as sns

class Func:
    def __init__(self):
        self.path_input= 'E:/节能减排/EP协同仿真文件打包/fit_function/fit_function/control.csv'
        self.path_cop='E:/节能减排/EP协同仿真文件打包/fit_function/fit_function/cop.csv'
        self.path_power = 'E:/节能减排/EP协同仿真文件打包/fit_function/fit_function/power.csv'

    def train_model(self):
        # 训练COP模型
        self.data_input = pd.read_csv(self.path_input)
        self.data_output = pd.read_csv(self.path_cop)
        self.mlp=MLPRegressor(hidden_layer_sizes=(500,100,50),learning_rate_init=0.0001)
        self.mlp.fit(self.data_input.values, self.data_output.values[:,0])
        # 训练总功率模型
        self.data_output_power = pd.read_csv(self.path_power)
        self.mlp_power = MLPRegressor(hidden_layer_sizes=(500, 100, 50), learning_rate_init=0.0001)
        self.mlp_power.fit(self.data_input.values, self.data_output_power.values[:, 0])

    def predict(self,x):
        y_pred=self.mlp.predict(x)
        total_power = self.mlp_power.predict(x)
        return y_pred,total_power

class GeneticAlgorithm:
    def __init__(
            self,
            model,  ###############
            pop_size: int = 100,
            bits_per_var: int = 16,                  # 每个变量所用的二进制位数
            num_vars: int = 4,                       # 变量个数，默认为4
            max_gen: int = 200,
            crossover_rate: float = 0.9,
            mutation_rate: float = 0.02,
            x_bounds: List[Tuple[float, float]] = None
    ):
        """
        参数:
            pop_size: 种群大小
            bits_per_var: 每个变量的二进制编码位数
            num_vars: 变量个数 (这里默认为4)
            max_gen: 最大迭代代数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            x_bounds: 各变量的取值范围列表，如 [(-1,2), (-1,2), (-1,2), (-1,2)]
        """
        self.pop_size = pop_size
        self.bits_per_var = bits_per_var
        self.num_vars = num_vars
        self.chromosome_length = bits_per_var * num_vars  # 总染色体长度，每bits_per_var位代表一个变量

        self.max_gen = max_gen
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.func=model     ################
        # self.func.train_model()   ################

        # 假设变量都在同一个区间 [(-1, 2), (-1, 2), (-1, 2), (-1, 2)]
        if x_bounds is None:
            x_bounds = [(-1, 2)] * num_vars
        self.x_bounds = x_bounds

        # 记录每代的最佳适应度
        self.best_fitness_history = []
        self.avg_fitness_history = []

    def initialize_population(self) -> np.ndarray:
        """初始化种群: 随机生成二进制矩阵 (pop_size, chromosome_length)"""    # 此时的chromosome_length为4个变量所对应的长度
        return np.random.randint(2, size=(self.pop_size, self.chromosome_length))

    def decode_chromosome(self, chromosome: np.ndarray) -> np.ndarray:
        """
        将整条染色体解码为多维实数向量 x (长度 = num_vars)。
        每个变量占 bits_per_var 位，对应 x_bounds[i] 区间。
        """
        """将单个个体的二进制染色体解码为实数"""
        x = []
        for i in range(self.num_vars):
            # 提取该变量对应的二进制切片——每16位进行一次切片，每4次切片代表一个个体
            start = i * self.bits_per_var
            end = start + self.bits_per_var
            gene_slice = chromosome[start:end]

            # 转为十进制
            decimal_val = int(''.join(map(str, gene_slice)), 2)

            # 区间映射
            x_min, x_max = self.x_bounds[i]
            real_val = x_min + decimal_val * (x_max - x_min) / (2 ** self.bits_per_var - 1)
            x.append(real_val)  # 每个个体包含4个变量，以一维向量的形式返回
        return np.array(x)

    def decode_population(self, population: np.ndarray) -> np.ndarray:
    # 解码整个种群，返回形状为(pop_size, num_vars)的数组
        decoded = np.zeros((population.shape[0], self.num_vars))
        for i in range(population.shape[0]):
         decoded[i] = self.decode_chromosome(population[i])
        return decoded

    def fitness_function(self, x: np.ndarray):
        """
        假设多变量适应度函数：f(x1,x2,x3,x4) = sum(xi*sin(10πxi)) + 2.0
        可以根据需要换成任意多变量目标函数
        """
        # x =[4个控制变量]--ndarray数据类型
        # x = x.reshape(1, -1)
        cop, total_power = self.func.predict(x)  # 获取COP和总功率
        fitness = 0.7 * cop - 0.3 * total_power  # 权重可调
        return cop

    def calculate_fitness(self, population: np.ndarray) -> np.ndarray:
        """计算种群中所有个体的适应度"""
        decoded = self.decode_population(population)
        # 使用mlp进行批量预测
        fitness = self.fitness_function(decoded)
        return fitness

    def roulette_selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """轮盘赌选择"""
        fitness = fitness - np.min(fitness) + 1e-6  # 确保所有适应度为正；有点类似Min-Max归一化思想
        probs = fitness / fitness.sum()             # 所有个体根据适应度计算得到的生存概率
        selected_idx = np.random.choice(            # 根据存活概率选择生存的个体的索引
            self.pop_size,
            size=self.pop_size,
            p=probs
        )
        return population[selected_idx]             # 返回根据适应度选择得到的存活个体

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """单点交叉"""
        if np.random.random() < self.crossover_rate:  # 增加随机性——随机选择交叉
            point = np.random.randint(1, self.chromosome_length)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()         # 若未交叉，则选择遗传自上一代

    def mutation(self, chromosome: np.ndarray) -> np.ndarray:
        """位变异"""
        for i in range(self.chromosome_length):
            if np.random.random() < self.mutation_rate:  # 增加随机性——随机选择变异
                chromosome[i] = 1 - chromosome[i]
        return chromosome                                # 未变异返回原基因编码

    def evolve(self) -> Tuple[np.ndarray, float]:
        """执行遗传算法"""
        population = self.initialize_population()        # 初始化种群

        for generation in range(self.max_gen):           # 执行种群迭代
            fitness = self.calculate_fitness(population) # 计算种群中个体的适应度

            self.best_fitness_history.append(np.max(fitness))   # 选择每一代的最佳适应度
            self.avg_fitness_history.append(np.mean(fitness))   # 计算每一代的平均适应度

            selected = self.roulette_selection(population, fitness) # 根据适应度选择这一代存活的个体
            new_population = []     # 新个体

            for i in range(0, self.pop_size, 2):
                parent1, parent2 = selected[i], selected[i + 1]
                child1, child2 = self.crossover(parent1, parent2)   # 根据上一代存活的个体产生下一代个体：
                                                                    # 有一定概率通过交叉产生下一代个体；无交叉则直接遗传自父代。
                child1 = self.mutation(child1)                      # 产生的个体有一定概率发生变异
                child2 = self.mutation(child2)                      # 不变异则返回原基因编码
                new_population.extend([child1, child2])             # 产生的下一代个体

            population = np.array(new_population)                   # 利用产生的新个体进行下一代循环

        # 最后一代寻找最优个体
        final_fitness = self.calculate_fitness(population)          # 最后一代种群各个体的适应度
        best_idx = np.argmax(final_fitness)                         # 最佳适应度个体的索引
        best_chromosome = population[best_idx]                      # 最佳个体对应的基因序列
        best_x = self.decode_chromosome(best_chromosome)            # 最佳个体计算得到的自变量实值
        best_y = final_fitness[best_idx]
        # self.plot_optimization_process(population)
        return best_x, best_y


class EPSimulation:
    def __init__(self):
        # 初始化EnergyPlus API相关变量
        self.api = EnergyPlusAPI()
        self.state = self.api.state_manager.new_state()

        # 初始化句柄和状态标志
        self.initialize_handles()

        # 设置文件路径
        self.setup_file_paths()

        # 计数器初始化
        self.count = 0

        # 读取控制方案
        self.load_control_scheme()

        # 在初始化时设置回调
        self.api.runtime.callback_begin_system_timestep_before_predictor(
            self.state, self._time_step_handler_wrapper
        )
        # 获取时间
        self.time=[]

        # 获取控制参数
        self.control=[]

        # 获取输出参数
        self.output=[]

        # 获取数据
        self.cop=[]
        self.total_power = []  # 总功率记录
        self.mf_chwp_list = []  # 冷冻水泵流量记录
        self.mf_cwp_list = []  # 冷却水泵流量记录
        self.tchw_in_list = []  # 冷冻水入口温度
        self.tchw_out_list = []  # 冷冻水出口温度
        self.tcw_in_list = []  # 冷却水入口温度
        self.tcw_out_list = []  # 冷却水出口温度

        # self.func = Func()  #######################
        # self.func.train_model() #####################
        with open(r'E:\节能减排\EP协同仿真文件打包\fit_function\model.mo', 'rb') as f:
            self.func = pickle.load(f)

    def initialize_handles(self):
        """初始化所有EnergyPlus变量和执行器句柄"""
        self.one_time = True
        # 环境变量句柄
        self.oa_t_handle = -1
        self.oa_twb_handle = -1

        # 系统参数句柄
        self.cload_handle = -1
        self.tchw_in_handle = -1
        self.tchw_out_handle = -1
        self.tcw_in_handle = -1
        self.tcw_out_handle = -1

        # 设备功率句柄
        self.pwr_chiller_handle = -1
        self.pwr_chwp_handle = -1
        self.pwr_cwp_handle = -1
        self.pwr_ct1_handle = -1
        self.pwr_ct2_handle = -1
        self.pwr_ct3_handle = -1
        self.pwr_ct4_handle = -1

        # 流量句柄
        self.mf_chwp_handle = -1
        self.mf_cwp_handle = -1

        # 调度控制句柄
        self.sch_tchw_handle = -1
        self.sch_tcw_handle = -1
        self.sch_mf_chwp_handle = -1
        self.sch_mf_cwp_handle = -1

        # 状态标志
        self.ok = True
        self.exception = None

    def setup_file_paths(self):
        """设置模拟所需的文件路径"""
        self.idf_file = 'realSys.exp232.idf'
        self.weather_file = r'WeatherData\CHN_Hubei.Wuhan.574940_CSWD.epw'
        self.control_file = 'controlsch_0804_to_0905.csv'

        # 设置命令行参数
        self.argv_list = [
            '-w', self.weather_file,
            '-d', 'myout',
            self.idf_file
        ]

    def load_control_scheme(self):
        """加载控制方案数据"""
        try:
            control_sch = pd.read_csv(self.control_file, header=None)
            self.control_list = np.array(control_sch)
        except Exception as e:
            print(f"Error loading control scheme: {e}")
            self.ok = False

    def _time_step_handler_wrapper(self, state):
        """包装time_step_handler以确保正确的self引用"""
        return self.time_step_handler(state)

    def get_handles(self):
        """获取所有需要的EnergyPlus变量和执行器句柄"""
        try:
            # 获取环境变量句柄
            self.oa_t_handle = self.api.exchange.get_variable_handle(
                self.state, "SITE OUTDOOR AIR DRYBULB TEMPERATURE", "ENVIRONMENT"
            )
            self.oa_twb_handle = self.api.exchange.get_variable_handle(
                self.state, "SITE OUTDOOR AIR WETBULB TEMPERATURE", "ENVIRONMENT"
            )
            self.cload_handle = self.api.exchange.get_variable_handle(
                self.state, "PLANT SUPPLY SIDE COOLING DEMAND RATE", "CHW_SYS CHILLED WATER LOOP"
            )
            self.tchw_in_handle = self.api.exchange.get_variable_handle(
                self.state, "Plant Supply Side Inlet Temperature", "CHW_SYS CHILLED WATER LOOP"
            )
            self.tchw_out_handle = self.api.exchange.get_variable_handle(
                self.state, "Plant Supply Side Outlet Temperature", "CHW_SYS CHILLED WATER LOOP"
            )
            self.tcw_in_handle = self.api.exchange.get_variable_handle(
                self.state, "Plant Supply Side Inlet Temperature", "CHW_SYS CONDENSER WATER LOOP"
            )
            self.tcw_out_handle = self.api.exchange.get_variable_handle(
                self.state, "Plant Supply Side Outlet Temperature", "CHW_SYS CONDENSER WATER LOOP"
            )
            self.pwr_chiller_handle = self.api.exchange.get_variable_handle(
                self.state, "Chiller Electricity Rate", "MACCHILLER"
            )
            self.chiller_cooling_rate_handle = self.api.exchange.get_variable_handle(
                self.state, "Chiller Evaporator Cooling Rate", "MACCHILLER"
            )
            self.mf_chwp_handle = self.api.exchange.get_variable_handle(
                self.state, "Pump Mass Flow Rate", "CHW_SYS CHW SUPPLY PUMP"
            )
            self.mf_cwp_handle = self.api.exchange.get_variable_handle(
                self.state, "Pump Mass Flow Rate", "CHW_SYS CNDW SUPPLY PUMP"
            )
            self.pwr_chwp_handle = self.api.exchange.get_variable_handle(
                self.state, "Pump Electricity Rate", "CHW_SYS CHW SUPPLY PUMP"
            )
            self.pwr_cwp_handle = self.api.exchange.get_variable_handle(
                self.state, "Pump Electricity Rate", "CHW_SYS CNDW SUPPLY PUMP"
            )
            self.pwr_ct1_handle = self.api.exchange.get_variable_handle(
                self.state, "Cooling Tower Fan Electricity Rate", "CT1"
            )
            self.pwr_ct2_handle = self.api.exchange.get_variable_handle(
                self.state, "Cooling Tower Fan Electricity Rate", "CT2"
            )
            self.pwr_ct3_handle = self.api.exchange.get_variable_handle(
                self.state, "Cooling Tower Fan Electricity Rate", "CT3"
            )
            self.pwr_ct4_handle = self.api.exchange.get_variable_handle(
                self.state, "Cooling Tower Fan Electricity Rate", "CT4"
            )
            self.sch_tchw_handle = self.api.exchange.get_actuator_handle(
                self.state, "Schedule:Constant", "Schedule Value", "chwSet"
            )
            self.sch_tcw_handle = self.api.exchange.get_actuator_handle(
                self.state, "Schedule:Constant", "Schedule Value", "cwSet"
            )
            self.sch_mf_chwp_handle = self.api.exchange.get_actuator_handle(
                self.state, "Schedule:Constant", "Schedule Value", "chwpSch"
            )
            self.sch_mf_cwp_handle = self.api.exchange.get_actuator_handle(
                self.state, "Schedule:Constant", "Schedule Value", "cwpSch"
            )

        except Exception as e:
            self.exception = e
            self.ok = False
            raise e

    def check_handles(self):
        """检查所有句柄是否正确获取"""
        return not (
                self.oa_t_handle == -1
                or self.pwr_chiller_handle == -1
                or self.cload_handle == -1
                or self.tchw_in_handle == -1
                or self.mf_chwp_handle == -1
                or self.mf_cwp_handle == -1
                or self.sch_tchw_handle == -1
                or self.sch_tcw_handle == -1
                or self.sch_mf_cwp_handle == -1
                or self.sch_mf_chwp_handle == -1
        )

    def time_step_handler(self, state):
        """时间步长处理函数"""
        sys.stdout.flush()

        # 检查API是否准备就绪
        if not self.api.exchange.api_data_fully_ready(state) or not self.ok:
            return

        # 首次运行时获取句柄
        if self.one_time:
            self.one_time = False
            self.get_handles()
            if not self.check_handles():
                self.ok = False
                return

        # 获取当前模拟时间
        sim_time = self.api.exchange.current_sim_time(state)
        # 添加年份检查
        epw_year = self.api.exchange.year(state)
        run_period_year = self.api.exchange.calendar_year(state)

        # 在预热期后开始控制
        if self.count > 719:
            self.apply_control(state, sim_time)
            self.time.append(sim_time)
        else:
            print(f"warm time is: {sim_time}")

        self.count += 1

    def apply_control(self, state, sim_time):
        """应用控制策略"""
        # 添加索引检查
        if self.count - 719 >= len(self.control_list):
            print("Warning: Control list index out of range")
            return

        ans = self.control_list[self.count - 719, :]

        try:
            # 读取运行参数
            oa_t = self.api.exchange.get_variable_value(state, self.oa_t_handle)
            oa_twb = self.api.exchange.get_variable_value(state, self.oa_twb_handle)
            cload = self.api.exchange.get_variable_value(state, self.cload_handle)
            tchw_in = self.api.exchange.get_variable_value(state, self.tchw_in_handle)
            tchw_out = self.api.exchange.get_variable_value(state, self.tchw_out_handle)
            tcw_in = self.api.exchange.get_variable_value(state, self.tcw_in_handle)
            tcw_out = self.api.exchange.get_variable_value(state, self.tcw_out_handle)
            pwr_chwp = self.api.exchange.get_variable_value(state, self.pwr_chwp_handle)
            pwr_cwp = self.api.exchange.get_variable_value(state, self.pwr_cwp_handle)
            pwr_ct1 = self.api.exchange.get_variable_value(state, self.pwr_ct1_handle)
            pwr_ct2 = self.api.exchange.get_variable_value(state, self.pwr_ct2_handle)
            pwr_ct3 = self.api.exchange.get_variable_value(state, self.pwr_ct3_handle)
            pwr_ct4 = self.api.exchange.get_variable_value(state, self.pwr_ct4_handle)
            mf_chwp = self.api.exchange.get_variable_value(state, self.mf_chwp_handle)
            mf_cwp = self.api.exchange.get_variable_value(state, self.mf_cwp_handle)
            chiller_cooling_rate=self.api.exchange.get_variable_value(state, self.chiller_cooling_rate_handle)
            pwr_chiller = self.api.exchange.get_variable_value(state, self.pwr_chiller_handle)
            total_power = pwr_chiller + pwr_chwp + pwr_cwp + pwr_ct1 + pwr_ct2 + pwr_ct3 + pwr_ct4
            self.total_power.append(total_power)  # 记录总功率
            self.mf_chwp_list.append(mf_chwp)
            self.mf_cwp_list.append(mf_cwp)
            self.tchw_in_list.append(tchw_in)
            self.tchw_out_list.append(tchw_out)
            self.tcw_in_list.append(tcw_in)
            self.tcw_out_list.append(tcw_out)

            if pwr_chiller > 1e-6:  # 避免除以零
                cop = chiller_cooling_rate / pwr_chiller
            else:
                cop = 0.0

            self.cop.append(cop)
            print('COP:',cop)
            print(f"第{self.count - 719}次计算COP，当前count值：{self.count}")
            # 遗传算法
            ga = GeneticAlgorithm(
                self.func,  ######################
                pop_size=200,
                bits_per_var=16,
                num_vars=4,
                max_gen=200,
                crossover_rate=0.9,
                mutation_rate=0.01,
                x_bounds=[(0.2, 1), (0.2, 1), (4.5,10), (30, 37)]  # 四个变量，取值范围都相同
            )
            # 执行算法
            best_x, best_fitness = ga.evolve()
            ans=best_x
            # 设置控制参数
            self.api.exchange.set_actuator_value(state, self.sch_mf_chwp_handle, float(ans[0]))
            self.api.exchange.set_actuator_value(state, self.sch_mf_cwp_handle, float(ans[1]))
            self.api.exchange.set_actuator_value(state, self.sch_tchw_handle, float(ans[2]))
            self.api.exchange.set_actuator_value(state, self.sch_tcw_handle, float(ans[3]))
            self.control.append(ans)

            print(f"count sim time is: {sim_time}", ans)

        except Exception as e:
            print(f"Error applying control values: {e}")
            self.ok = False

    def run_simulation(self):
        """运行EnergyPlus模拟"""
        # 请求变量
        self.api.exchange.request_variable(
            self.state, "SITE OUTDOOR AIR DRYBULB TEMPERATURE", "ENVIRONMENT"
        )
        self.api.exchange.request_variable(
            self.state, "SITE OUTDOOR AIR DEWPOINT TEMPERATURE", "ENVIRONMENT"
        )

        # 添加错误处理
        try:
            return_code = self.api.runtime.run_energyplus(self.state, self.argv_list)
            return return_code
        except Exception as e:
            print(f"Simulation failed: {e}")
            return -1


def main():
    """主函数"""
    # 运行遗传算法优化策略
    # simulation_ga = EPSimulation()
    # return_code_ga = simulation_ga.run_simulation()
    # print(simulation_ga.count - 720)
    # print(f"Simulation completed with return code: {return_code_ga}")
    #
    # df_ga = pd.DataFrame({
    #     'Timestamp': simulation_ga.time,
    #     'ChW_Pump_Flow': simulation_ga.mf_chwp_list,
    #     'CndW_Pump_Flow': simulation_ga.mf_cwp_list,
    #     'ChW_In_Temp': simulation_ga.tchw_in_list,
    #     'ChW_Out_Temp': simulation_ga.tchw_out_list,
    #     'CndW_In_Temp': simulation_ga.tcw_in_list,
    #     'CndW_Out_Temp': simulation_ga.tcw_out_list,
    #     'COP': simulation_ga.cop,
    #     'Total_Power': simulation_ga.total_power
    # })
    # df_ga.to_csv('optimized_data.csv', index=False)

    # 读取数据
    df_ga = pd.read_csv('optimized_data.csv')
    df_fixed = pd.read_csv('baseline_data.csv')

    # 计算平均COP提升
    avg_cop_ga = df_ga['COP'].mean()
    print(f"COP: {avg_cop_ga}")
    avg_cop_fixed = df_fixed['COP'].mean()
    print(f"COP: {avg_cop_fixed}")
    cop_improvement = (avg_cop_ga - avg_cop_fixed) / avg_cop_fixed * 100

    #计算总节能量（假设时间步长为1分钟，Δt=1/60小时）
    total_energy_ga = df_ga['Total_Power'].sum() * (1 / 60) * 0.001  # 单位：kWh
    print(f"Total Energy: {total_energy_ga}")
    total_energy_fixed = df_fixed['Total_Power'].sum() * (1 / 60) * 0.001
    print(f"Total Energy: {total_energy_fixed}")
    energy_saving = total_energy_fixed - total_energy_ga
    energy_improvement = energy_saving / total_energy_fixed


    print(f"平均COP提升: {cop_improvement:.2f}%")
    print(f"总节能量: {energy_saving:.2f} kWh")
    print(f"总节能量提升: {energy_improvement*100:.2f}%")

    # 设置支持中文的字体
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    # plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # COP对比曲线
    # plt.figure(figsize=(12, 6))
    # plt.plot(df_ga['Timestamp'], df_ga['COP'], label='优化策略 (GA)', alpha=0.7)
    # plt.plot(df_fixed['Timestamp'], df_fixed['COP'], label='固定参数', linestyle='--', alpha=0.7)
    # plt.xlabel('时间')
    # plt.ylabel('COP')
    # plt.title('COP对比曲线')
    # plt.legend()
    # plt.savefig('cop_comparison.png')
    # plt.close()

    # 能耗对比柱状图
    # plt.figure(figsize=(8, 5))
    # sns.barplot(x=['优化策略', '固定参数'], y=[total_energy_ga, total_energy_fixed])
    # plt.ylabel('总能耗 (kWh)')
    # plt.title('总能耗对比')
    # plt.savefig('energy_comparison.png')
    # plt.close()

if __name__ == "__main__":
    main()