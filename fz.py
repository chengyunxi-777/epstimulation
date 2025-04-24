import sys
import os
import pandas as pd
from PyQt6 import uic
from PyQt6.QtWidgets import (QApplication, QMainWindow, QFileDialog,
                             QMessageBox, QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from EP_control_test_modified import EPSimulation
import traceback

class SimulationThread(QThread):
    update_log = pyqtSignal(str)
    simulation_finished = pyqtSignal(int)
    data_ready = pyqtSignal()

    def __init__(self, idf_path, weather_path, control_path, output_dir):
        super().__init__()
        self.idf_path = idf_path
        self.weather_path = weather_path
        self.control_path = control_path
        self.output_dir = output_dir
        self.simulation = None
        self.results = {}

    def run(self):
        try:
            # 初始化仿真
            self.update_log.emit("正在初始化仿真...")
            # 传递 output_dir 到 EPSimulation
            self.simulation = EPSimulation(
                self.idf_path,
                self.weather_path,
                self.control_path,
                self.output_dir  # 新增参数
            )

            # 修改仿真类的文件路径设置
            self.simulation.idf_file = self.idf_path
            self.simulation.weather_file = self.weather_path
            self.simulation.control_file = self.control_path
            self.simulation.argv_list = [
                '-w', self.weather_path,
                '-d', self.output_dir,  # 直接使用用户选择的目录
                self.idf_path
            ]
            self.simulation.output_dir = self.output_dir  # 传递输出目录给仿真类
            # 运行仿真
            self.update_log.emit("开始模拟运行...")
            return_code = self.simulation.run_simulation()

            # 保存结果
            self.update_log.emit("正在保存结果...")
            self.save_simulation_data()

            self.simulation_finished.emit(return_code)
            self.data_ready.emit()


        except Exception as e:

            self.update_log.emit(f"错误发生: {str(e)}\n{traceback.format_exc()}")  # 记录堆栈信息

            self.simulation_finished.emit(-1)
    def save_simulation_data(self):
        """保存数据到指定目录"""
        myout_dir = os.path.join(self.output_dir, "myout")
        fit_function_dir = os.path.join(self.output_dir, "fit_function", "fit_function")

        os.makedirs(myout_dir, exist_ok=True)
        os.makedirs(fit_function_dir, exist_ok=True)

        # 保存各文件
        pd.Series(self.simulation.time, name='time').to_csv(
            os.path.join(myout_dir, 'time.csv'), index=False)
        pd.DataFrame(self.simulation.control,
                     columns=['sch_mf_chwp_handle', 'sch_mf_cwp_handle', 'sch_tchw_handle', 'sch_tcw_handle']
                     ).to_csv(os.path.join(fit_function_dir, 'control.csv'), index=False)
        pd.DataFrame(self.simulation.output,
                     columns=['pwr_chiller', 'pwr_chwp', 'pwr_cwp', 'pwr_ct1', 'pwr_ct2', 'pwr_ct3']
                     ).to_csv(os.path.join(myout_dir, 'output.csv'), index=False)
        pd.Series(self.simulation.cop, name='COP').to_csv(
            os.path.join(fit_function_dir, 'cop.csv'), index=False)
        pd.Series(self.simulation.total_power, name='Total_Power').to_csv(
            os.path.join(fit_function_dir, 'power.csv'), index=False)

        # 收集结果路径供表格显示
        self.results = {
            'baseline_data': os.path.join(myout_dir, 'baseline_data.csv'),
            'output': os.path.join(myout_dir, 'output.csv')
        }
        df = pd.DataFrame({
            'Timestamp': self.simulation.time,
            'ChW_Pump_Flow': self.simulation.mf_chwp_list,
            'CndW_Pump_Flow': self.simulation.mf_cwp_list,
            'ChW_In_Temp': self.simulation.tchw_in_list,
            'ChW_Out_Temp': self.simulation.tchw_out_list,
            'CndW_In_Temp': self.simulation.tcw_in_list,
            'CndW_Out_Temp': self.simulation.tcw_out_list,
            'COP': self.simulation.cop,
            'Total_Power': self.simulation.total_power
        })
        df.to_csv(os.path.join(myout_dir, 'baseline_data.csv'), index=False)
        # =====================================

        # 收集结果路径供表格显示
        self.results = {
            'baseline_data': os.path.join(myout_dir, 'baseline_data.csv'),
            'output': os.path.join(myout_dir, 'output.csv')
        }

    class MainWindow(QMainWindow):
        # ... 其他代码不变 ...

        def load_results_to_table(self):
            """将结果加载到表格"""
            try:
                # 读取基准数据
                df = pd.read_csv(self.simulation_thread.results['baseline_data'])

                # ==== 优化：清空表格并动态设置列 ====
                self.tableWidget_baseline.clearContents()
                self.tableWidget_baseline.setRowCount(len(df))
                self.tableWidget_baseline.setColumnCount(len(df.columns))
                self.tableWidget_baseline.setHorizontalHeaderLabels(df.columns)

                for row in range(len(df)):
                    for col in range(len(df.columns)):
                        item = QTableWidgetItem(str(df.iat[row, col]))
                        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                        self.tableWidget_baseline.setItem(row, col, item)
                # =====================================

                self.update_log("数据已成功加载到表格")
            except Exception as e:
                self.update_log(f"加载数据到表格失败：{str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 动态加载UI文件
        uic.loadUi("fz.ui", self)

        # 初始化UI状态
        self.simulation_thread = None
        self.setup_connections()
        self.init_table()

    def init_table(self):
        """初始化基准数据表格"""
        self.tableWidget_baseline.setColumnCount(8)
        self.tableWidget_baseline.setHorizontalHeaderLabels([
            "时间", "冷冻水泵流量", "冷却水泵流量",
            "冷冻水入口温度", "冷冻水出口温度",
            "冷却水入口温度", "冷却水出口温度", "COP"
        ])
        self.tableWidget_baseline.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def setup_connections(self):
        """连接信号与槽"""
        self.pushButton_browse_idf.clicked.connect(lambda: self.select_file("IDF文件 (*.idf)", self.lineEdit_idf))
        self.pushButton_browse_weather.clicked.connect(
            lambda: self.select_file("气象文件 (*.epw)", self.lineEdit_weather))
        self.pushButton_browse_control.clicked.connect(
            lambda: self.select_file("控制文件 (*.csv)", self.lineEdit_control))
        self.pushButton_run_baseline.clicked.connect(self.start_simulation)

    def select_file(self, file_filter, line_edit):
        """文件选择通用方法"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", file_filter)
        if file_path:
            line_edit.setText(file_path)

    def start_simulation(self):
        """启动仿真线程"""
        if not self.validate_inputs():
            return

        # 选择输出目录
        output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if not output_dir:
            return
        # 确保路径使用正斜杠，避免转义问题
        output_dir = output_dir.replace("\\", "/")  # 新增代码
        # 创建并启动线程
        self.simulation_thread = SimulationThread(
            self.lineEdit_idf.text(),
            self.lineEdit_weather.text(),
            self.lineEdit_control.text(),
            output_dir
        )

        # 连接信号
        self.simulation_thread.update_log.connect(self.update_log)
        self.simulation_thread.simulation_finished.connect(self.on_simulation_finished)
        self.simulation_thread.data_ready.connect(self.load_results_to_table)

        # 禁用按钮防止重复点击
        self.pushButton_run_baseline.setEnabled(False)
        self.simulation_thread.start()

    def validate_inputs(self):
        """验证输入文件是否存在"""
        missing = []
        if not os.path.exists(self.lineEdit_idf.text()):
            missing.append("IDF文件")
        if not os.path.exists(self.lineEdit_weather.text()):
            missing.append("气象文件")
        if not os.path.exists(self.lineEdit_control.text()):
            missing.append("控制文件")

        if missing:
            QMessageBox.critical(self, "错误", f"以下文件不存在：\n{', '.join(missing)}")
            return False
        return True

    def update_log(self, message):
        """实时更新日志"""
        self.textBrowser_log.append(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {message}")
        QApplication.processEvents()  # 强制刷新UI

    def on_simulation_finished(self, return_code):
        """仿真完成处理"""
        self.pushButton_run_baseline.setEnabled(True)
        if return_code == 0:
            QMessageBox.information(self, "完成", "仿真成功完成！")
        else:
            QMessageBox.critical(self, "错误", f"仿真异常结束，返回值：{return_code}")

    def load_results_to_table(self):
        """将结果加载到表格"""
        try:
            # 读取基准数据
            df = pd.read_csv(self.simulation_thread.results['baseline_data'])

            # 设置表格
            self.tableWidget_baseline.setRowCount(len(df))
            for row in range(len(df)):
                for col in range(8):
                    item = QTableWidgetItem(str(df.iat[row, col]))
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    self.tableWidget_baseline.setItem(row, col, item)

            self.update_log("数据已成功加载到表格")

        except Exception as e:
            self.update_log(f"加载数据到表格失败：{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())