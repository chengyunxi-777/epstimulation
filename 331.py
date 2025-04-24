import sys
import pickle
import pandas as pd
from datetime import datetime
from PyQt6.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QHeaderView, QFileDialog
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PyQt6 import uic
from 优化算法 import GeneticAlgorithm, EPSimulation,Func


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 动态加载 UI 文件
        uic.loadUi("fz.ui", self)
        self.output_path = ""  # 添加存储路径变量
        # 初始化界面
        self.setup_ui()
        self.setup_connections()

        # 加载机器学习模型
        try:
            with open(r'E:\节能减排\EP协同仿真文件打包\fit_function\model.mo', 'rb') as f:
                self.model = pickle.load(f)  # 现在可以正确加载
        except FileNotFoundError:
            self.update_log("❌ 错误：未找到机器学习模型文件。")
        except Exception as e:
            self.update_log(f"❌ 加载模型时发生错误：{str(e)}")

        self.worker = None
        self.thread = None

    def setup_ui(self):
        # 配置表格
        self.tableWidget_optimized.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tableWidget_optimized.setColumnCount(9)
        self.tableWidget_optimized.setHorizontalHeaderLabels([
            "时间戳", "冷冻水流量", "冷却水流量",
            "冷冻水入口温度", "冷冻水出口温度",
            "冷却水入口温度", "冷却水出口温度",
            "COP", "总功率"
        ])

        # 设置进度条
        self.progressBar_ga.setRange(0, 100)
        self.progressBar_ga.setValue(0)

    def setup_connections(self):
        self.pushButton_run_optimized.clicked.connect(self.toggle_simulation)

    def toggle_simulation(self):
        if self.worker and hasattr(self.worker, 'isRunning') and self.worker.isRunning():
            self.stop_simulation()
        else:
            self.start_simulation()

    def start_simulation(self):
        # 添加文件选择对话框
        self.output_path, _ = QFileDialog.getSaveFileName(
            self,
            "选择保存路径",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )

        if not self.output_path:  # 用户取消选择
            self.update_log("⚠️ 已取消操作：未选择保存路径")
            return

        # 获取参数
        params = {
            'pop_size': self.spinBox_pop_size.value(),
            'max_gen': self.spinBox_max_gen.value(),
            'crossover_rate': self.doubleSpinBox_crossover.value(),
            'mutation_rate': self.doubleSpinBox_mutation.value()
        }

        var_ranges = [
            tuple(map(float, self.lineEdit_var1_range.text().split(','))),
            tuple(map(float, self.lineEdit_var2_range.text().split(','))),
            tuple(map(float, self.lineEdit_var3_range.text().split(','))),
            tuple(map(float, self.lineEdit_var4_range.text().split(',')))
        ]

        # 创建线程
        self.thread = QThread()
        self.worker = SimulationWorker(self.model, params, var_ranges, self.output_path)
        self.worker.moveToThread(self.thread)

        # 连接信号
        self.worker.update_table.connect(self.update_table)
        self.worker.update_log.connect(self.update_log)
        self.worker.update_progress.connect(self.update_progress)
        self.worker.finished.connect(self.thread.quit)
        self.thread.started.connect(self.worker.run)
        self.thread.finished.connect(self.on_finished)

        # 清空旧数据
        self.tableWidget_optimized.setRowCount(0)
        self.textBrowser_opt.clear()

        # 启动线程
        self.thread.start()
        self.pushButton_run_optimized.setText("停止仿真")

    def stop_simulation(self):
        if self.worker:
            self.worker.stop()
        self.pushButton_run_optimized.setText("开始优化仿真")

    def on_finished(self):
        self.progressBar_ga.setValue(100)
        self.pushButton_run_optimized.setText("开始优化仿真")

    def update_table(self, data):
        row = self.tableWidget_optimized.rowCount()
        self.tableWidget_optimized.insertRow(row)

        self.tableWidget_optimized.setItem(row, 0, QTableWidgetItem(data['Timestamp']))
        self.tableWidget_optimized.setItem(row, 1, QTableWidgetItem(f"{data['ChW_Pump_Flow']:.2f}"))
        self.tableWidget_optimized.setItem(row, 2, QTableWidgetItem(f"{data['CndW_Pump_Flow']:.2f}"))
        self.tableWidget_optimized.setItem(row, 3, QTableWidgetItem(f"{data['ChW_In_Temp']:.1f}"))
        self.tableWidget_optimized.setItem(row, 4, QTableWidgetItem(f"{data['ChW_Out_Temp']:.1f}"))
        self.tableWidget_optimized.setItem(row, 5, QTableWidgetItem(f"{data['CndW_In_Temp']:.1f}"))
        self.tableWidget_optimized.setItem(row, 6, QTableWidgetItem(f"{data['CndW_Out_Temp']:.1f}"))
        self.tableWidget_optimized.setItem(row, 7, QTableWidgetItem(f"{data['COP']:.2f}"))
        self.tableWidget_optimized.setItem(row, 8, QTableWidgetItem(f"{data['Total_Power']:.0f}"))

        # 自动滚动到最后一行
        self.tableWidget_optimized.scrollToBottom()

    def update_log(self, message):
        self.textBrowser_opt.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        self.textBrowser_opt.verticalScrollBar().setValue(
            self.textBrowser_opt.verticalScrollBar().maximum()
        )

    def update_progress(self, value):
        self.progressBar_ga.setValue(value)


class SimulationWorker(QObject):
    update_table = pyqtSignal(dict)
    update_log = pyqtSignal(str)
    update_progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, model, params, var_ranges, output_path):
        super().__init__()
        self.model = model
        self.params = params
        self.var_ranges = var_ranges
        self._is_running = True
        self.output_path = output_path

    def run(self):
        try:
            # 初始化遗传算法
            ga = GeneticAlgorithm(
                self.model,
                pop_size=self.params['pop_size'],
                crossover_rate=self.params['crossover_rate'],
                mutation_rate=self.params['mutation_rate'],
                max_gen=self.params['max_gen'],
                x_bounds=self.var_ranges
            )

            # 运行遗传算法优化
            best_x, best_y = ga.evolve()
            self.update_log.emit(f"🎉 优化完成！最佳适应度：{best_y:.2f}")

            # 初始化 EnergyPlus 仿真
            self.update_log.emit("🚀 开始 EnergyPlus 仿真...")
            simulator = EPSimulation()  # 根据实际情况初始化参数
            return_code = simulator.run_simulation()

            # 实时数据更新
            for idx, (time, flow, power) in enumerate(zip(
                    simulator.time,
                    simulator.mf_chwp_list,
                    simulator.total_power
            )):
                if not self._is_running:
                    break

                data = {
                    'Timestamp': datetime.strftime(time, "%Y-%m-%d %H:%M"),
                    'ChW_Pump_Flow': flow,
                    'CndW_Pump_Flow': simulator.mf_cwp_list[idx],
                    'ChW_In_Temp': simulator.tchw_in_list[idx],
                    'ChW_Out_Temp': simulator.tchw_out_list[idx],
                    'CndW_In_Temp': simulator.tcw_in_list[idx],
                    'CndW_Out_Temp': simulator.tcw_out_list[idx],
                    'COP': simulator.cop[idx],
                    'Total_Power': power
                }
                self.update_table.emit(data)
                self.update_progress.emit(int((idx + 1) / len(simulator.time) * 100))

            self.update_log.emit(f"✅ 仿真完成！返回码：{return_code}")

        except Exception as e:
            self.update_log.emit(f"❌ 发生错误：{str(e)}")
        finally:
            try:
                df = pd.DataFrame({
                    'Timestamp': simulator.time,
                    'ChW_Pump_Flow': simulator.mf_chwp_list,
                    'CndW_Pump_Flow': simulator.mf_cwp_list,
                    'ChW_In_Temp': simulator.tchw_in_list,
                    'ChW_Out_Temp': simulator.tchw_out_list,
                    'CndW_In_Temp': simulator.tcw_in_list,
                    'CndW_Out_Temp': simulator.tcw_out_list,
                    'COP': simulator.cop,
                    'Total_Power': simulator.total_power
                })
                df.to_csv(self.output_path, index=False)
                self.update_log.emit(f"✅ 数据已保存至：{self.output_path}")
            except Exception as e:
                self.update_log.emit(f"❌ 保存数据失败：{str(e)}")
            self.finished.emit()

    def stop(self):
        self._is_running = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
