# main.py
import sys
import os
import traceback
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog,
    QMessageBox, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6 import uic
from 优化算法 import EPSimulation, GeneticAlgorithm,Func


class OptimizationThread(QThread):
    update_log = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    simulation_finished = pyqtSignal(int)
    data_ready = pyqtSignal()

    def __init__(self, config):
        super().__init__()
        self.idf_path = config['idf_path']
        self.weather_path = config['weather_path']
        self.control_path = config['control_path']
        self.output_dir = config['output_dir']
        self.ga_params = config['ga_params']
        self.var_ranges = config['var_ranges']
        self.simulation = None

    def run(self):
        try:
            self.update_log.emit("初始化优化仿真...")

            # 配置EnergyPlus参数
            self.simulation = EPSimulation()
            self.simulation.idf_file = self.idf_path
            self.simulation.weather_file = self.weather_path
            self.simulation.control_file = self.control_path
            self.simulation.argv_list = [
                '-w', self.weather_path,
                '-d', self.output_dir,
                self.idf_path
            ]

            # 配置遗传算法参数
            self.simulation.ga_config = {
                'pop_size': self.ga_params['pop_size'],
                'max_gen': self.ga_params['max_gen'],
                'crossover_rate': self.ga_params['crossover_rate'],
                'mutation_rate': self.ga_params['mutation_rate'],
                'var_ranges': self._parse_var_ranges()
            }

            self.update_log.emit("启动优化进程...")
            return_code = self.simulation.run_simulation()

            if return_code == 0:
                self._save_optimized_data()
                self.data_ready.emit()

            self.simulation_finished.emit(return_code)

        except Exception as e:
            self.update_log.emit(f"仿真错误: {str(e)}\n{traceback.format_exc()}")
            self.simulation_finished.emit(-1)

    def _parse_var_ranges(self):
        """解析变量范围输入"""
        ranges = []
        for var_range in self.var_ranges:
            try:
                min_val, max_val = map(float, var_range.split(','))
                ranges.append((min_val, max_val))
            except:
                ranges.append((0.0, 1.0))  # 默认值
        return ranges

    def _save_optimized_data(self):
        """保存优化结果"""
        output_path = os.path.join(self.output_dir, "optimized_results.csv")
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
        df.to_csv(output_path, index=False)
        self.update_log.emit(f"结果已保存至: {output_path}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("fz.ui", self)

        # 初始化UI状态
        self.output_dir = ""
        self.optimization_thread = None
        self._setup_ui()
        self._setup_connections()

    def _setup_ui(self):
        """初始化界面组件"""
        # 设置表格属性
        for table in [self.tableWidget_baseline, self.tableWidget_optimized]:
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            table.setEditTriggers(table.EditTrigger.NoEditTriggers)

        # 初始化遗传算法参数
        self.spinBox_pop_size.setValue(200)
        self.spinBox_max_gen.setValue(100)
        self.doubleSpinBox_crossover.setValue(0.9)
        self.doubleSpinBox_mutation.setValue(0.01)

        # 初始化变量范围
        default_ranges = ["0.2,1", "0.2,1", "4.5,10", "30,37"]
        self.lineEdit_var1_range.setText(default_ranges[0])
        self.lineEdit_var2_range.setText(default_ranges[1])
        self.lineEdit_var3_range.setText(default_ranges[2])
        self.lineEdit_var4_range.setText(default_ranges[3])

    def _setup_connections(self):
        """连接信号与槽"""
        # 文件选择
        self.pushButton_browse_idf.clicked.connect(
            lambda: self._select_file("IDF文件 (*.idf)", self.lineEdit_idf))
        self.pushButton_browse_weather.clicked.connect(
            lambda: self._select_file("气象文件 (*.epw)", self.lineEdit_weather))
        self.pushButton_browse_control.clicked.connect(
            lambda: self._select_file("控制文件 (*.csv)", self.lineEdit_control))

        # 优化仿真按钮
        self.pushButton_run_optimized.clicked.connect(self._start_optimization)

    def _select_file(self, filter_str, line_edit):
        """通用文件选择方法"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", filter_str)
        if file_path:
            line_edit.setText(file_path.replace("/", "\\"))

    def _validate_inputs(self):
        """验证输入有效性"""
        missing = []
        for widget, name in [
            (self.lineEdit_idf, "IDF文件"),
            (self.lineEdit_weather, "气象文件"),
            (self.lineEdit_control, "控制文件")
        ]:
            if not os.path.exists(widget.text()):
                missing.append(name)

        if missing:
            QMessageBox.critical(self, "错误", f"缺失必要文件:\n{', '.join(missing)}")
            return False
        return True

    def _start_optimization(self):
        """启动优化仿真"""
        if not self._validate_inputs():
            return

        # 选择输出目录
        output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if not output_dir:
            return
        self.output_dir = output_dir.replace("\\", "/")

        # 获取遗传算法参数
        ga_params = {
            'pop_size': self.spinBox_pop_size.value(),
            'max_gen': self.spinBox_max_gen.value(),
            'crossover_rate': self.doubleSpinBox_crossover.value(),
            'mutation_rate': self.doubleSpinBox_mutation.value()
        }

        # 获取变量范围
        var_ranges = [
            self.lineEdit_var1_range.text(),
            self.lineEdit_var2_range.text(),
            self.lineEdit_var3_range.text(),
            self.lineEdit_var4_range.text()
        ]

        # 创建配置字典
        config = {
            'idf_path': self.lineEdit_idf.text(),
            'weather_path': self.lineEdit_weather.text(),
            'control_path': self.lineEdit_control.text(),
            'output_dir': self.output_dir,
            'ga_params': ga_params,
            'var_ranges': var_ranges
        }

        # 创建并启动线程
        self.optimization_thread = OptimizationThread(config)
        self.optimization_thread.update_log.connect(self._update_log)
        self.optimization_thread.progress_updated.connect(self._update_progress)
        self.optimization_thread.simulation_finished.connect(self._on_simulation_finished)
        self.optimization_thread.data_ready.connect(self._load_results)

        self.pushButton_run_optimized.setEnabled(False)
        self.optimization_thread.start()

    def _update_log(self, message):
        """更新日志浏览器"""
        self.textBrowser_opt.append(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {message}")

    def _update_progress(self, value):
        """更新进度条"""
        self.progressBar_ga.setValue(value)

    def _on_simulation_finished(self, return_code):
        """仿真完成处理"""
        self.pushButton_run_optimized.setEnabled(True)
        if return_code == 0:
            QMessageBox.information(self, "完成", "优化仿真成功完成!")
        else:
            QMessageBox.critical(self, "错误", f"仿真异常终止，代码: {return_code}")

    def _load_results(self):
        """加载结果到表格"""
        try:
            csv_path = os.path.join(self.output_dir, "optimized_results.csv")
            df = pd.read_csv(csv_path)

            self.tableWidget_optimized.setRowCount(len(df))
            self.tableWidget_optimized.setColumnCount(len(df.columns))
            self.tableWidget_optimized.setHorizontalHeaderLabels(df.columns)

            for row_idx, row in df.iterrows():
                for col_idx, value in enumerate(row):
                    item = QTableWidgetItem(str(value))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.tableWidget_optimized.setItem(row_idx, col_idx, item)

            self._update_log("优化结果已加载")
        except Exception as e:
            self._update_log(f"加载失败: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())