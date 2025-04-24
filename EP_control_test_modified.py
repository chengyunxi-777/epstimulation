import sys
from cProfile import label

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from deap.benchmarks.tools import scipy_imported

# 添加EnergyPlus路径
sys.path.append('F:/energyplus23.2')
from pyenergyplus.api import EnergyPlusAPI

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

            #输出参数
            self.output.append([pwr_chiller,pwr_chwp,pwr_cwp,pwr_ct1,pwr_ct2,pwr_ct3])
            print('chiller_cooling_rate:', chiller_cooling_rate)
            print('pwr_chiller:', pwr_chiller)

            if pwr_chiller > 1e-6:  # 避免除以零
                cop = chiller_cooling_rate / pwr_chiller
            else:
                cop = 0.0

            self.cop.append(cop)
            print('COP:',cop)
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
    simulation = EPSimulation()
    return_code = simulation.run_simulation()
    df = pd.DataFrame({
        'Timestamp': simulation.time,
        'ChW_Pump_Flow': simulation.mf_chwp_list,
        'CndW_Pump_Flow': simulation.mf_cwp_list,
        'ChW_In_Temp': simulation.tchw_in_list,
        'ChW_Out_Temp': simulation.tchw_out_list,
        'CndW_In_Temp': simulation.tcw_in_list,
        'CndW_Out_Temp': simulation.tcw_out_list,
        'COP': simulation.cop,
        'Total_Power': simulation.total_power
    })
    df.to_csv('baseline_data.csv', index=False)
    print(f"Simulation completed with return code: {return_code}")
    pd.Series(simulation.time,name='time').to_csv('E:/节能减排/EP协同仿真文件打包/myout/time.csv',index=False)
    pd.DataFrame(simulation.control,columns=['sch_mf_chwp_handle','sch_mf_cwp_handle','sch_tchw_handle','sch_tcw_handle']).to_csv(
        'E:/节能减排/EP协同仿真文件打包/fit_function/fit_function/control.csv', index=False)
    pd.DataFrame(simulation.output,columns=['pwr_chiller','pwr_chwp','pwr_cwp','pwr_ct1','pwr_ct2','pwr_ct3']).to_csv('E:/节能减排/EP协同仿真文件打包/myout/output.csv',index=False)
    pd.Series(simulation.cop,name='COP').to_csv('E:/节能减排/EP协同仿真文件打包/fit_function/fit_function/cop.csv',index=False)
    pd.Series(simulation.total_power, name='Total_Power').to_csv(
        'E:/节能减排/EP协同仿真文件打包/fit_function/fit_function/power.csv', index=False)
    print(simulation.count-720)
    print(pd.DataFrame(simulation.control).shape)

if __name__ == "__main__":
    main()