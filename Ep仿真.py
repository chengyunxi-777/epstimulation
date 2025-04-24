from PyQt6.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QTextEdit, QStackedWidget, QComboBox, QWidget, \
    QPushButton, QFileDialog, QMessageBox, QLineEdit
from PyQt6 import uic
from PyQt6.QtGui import QIcon  # 关键修正：单独导入QIcon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.ticker import MultipleLocator
import pandas as pd
import numpy as np
import logging
import seaborn as sns
import sys
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义颜色列表
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
          'tab:olive', 'tab:cyan']


def resource_path(relative_path):
    """ 根据打包环境适配资源路径 """
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 设置窗口图标（使用正确的QIcon导入）
        self.setWindowIcon(QIcon(resource_path("title.ico")))  # 关键修正

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 加载UI文件
        uic.loadUi(resource_path("Ep.ui"), self)

        # 初始化文件选择控件
        self.btn_select_base = self.findChild(QPushButton, "btn_select_base")
        self.btn_select_opt = self.findChild(QPushButton, "btn_select_opt")
        self.lineEdit_base = self.findChild(QLineEdit, "lineEdit_base")
        self.lineEdit_opt = self.findChild(QLineEdit, "lineEdit_opt")

        # 绑定事件
        self.btn_select_base.clicked.connect(lambda: self.select_file('base'))
        self.btn_select_opt.clicked.connect(lambda: self.select_file('opt'))

        # 初始化数据容器
        self.df_fixed = None  # 基础数据
        self.df_ga = None  # 优化数据

        # 建立选项映射关系（保持不变）
        self.option_mapping = {
            "冷冻水泵流量_base": "ChW_Pump_Flow",
            "冷却水泵流量_base": "CndW_Pump_Flow",
            "冷冻水出水温度_base": "ChW_Out_Temp",
            "冷却水出水温度_base": "CndW_Out_Temp",
            "COP_base": "COP",
            "总功率_base": "Total_Power"
        }
        self.option_mapping_opt = {
            "冷冻水泵流量_opt": "ChW_Pump_Flow",
            "冷却水泵流量_opt": "CndW_Pump_Flow",
            "冷冻水出水温度_opt": "ChW_Out_Temp",
            "冷却水出水温度_opt": "CndW_Out_Temp",
            "COP_opt": "COP",
            "总功率_opt": "Total_Power"
        }

        # 初始化图表组件（保持不变）
        self.init_matplotlib_widgets()
        self.setup_combobox_connections()
        self.init_comparison_widgets()

        # 初始化页面切换功能（保持不变）
        self.textEdit = self.findChild(QTextEdit, "textEdit")
        self.stacked_widget = self.findChild(QStackedWidget, "stackedWidget")
        self.stacked_widget_2 = self.findChild(QStackedWidget, "stackedWidget_2")
        self.btn_page1()
        self.btn_page2()

        # 绑定导出按钮（保持不变）
        self.pushButton_3 = self.findChild(QPushButton, "pushButton_3")
        self.pushButton_3.clicked.connect(self.export_report)


    def init_matplotlib_widgets(self):
        # 初始化所有图表容器
        self.widget_base1 = self.findChild(QWidget, "widget_base1")
        self.widget_base2 = self.findChild(QWidget, "widget_base2")
        self.widget_opt1 = self.findChild(QWidget, "widget_opt1")
        self.widget_opt2 = self.findChild(QWidget, "widget_opt2")
        self.widget_COP = self.findChild(QWidget, "widget_COP")
        self.widget_E = self.findChild(QWidget, "widget_E")

        # 创建带工具栏的图表
        self.canvas_mapping = {
            "widget_base1": self.create_canvas_with_toolbar(self.widget_base1),
            "widget_base2": self.create_canvas_with_toolbar(self.widget_base2),
            "widget_opt1": self.create_canvas_with_toolbar(self.widget_opt1),
            "widget_opt2": self.create_canvas_with_toolbar(self.widget_opt2),
            "widget_COP": self.create_canvas_with_toolbar(self.widget_COP),
            "widget_E": self.create_canvas_with_toolbar(self.widget_E)
        }

    def init_comparison_widgets(self):
        """初始化对比分析图表布局"""
        # COP图表设置
        ax_cop, _, _ = self.canvas_mapping["widget_COP"]
        ax_cop.clear()
        ax_cop.set_xlabel('时间', fontsize=10)
        ax_cop.set_ylabel('COP', fontsize=10)
        ax_cop.xaxis.set_major_locator(MultipleLocator(30))

        # 能耗图表设置
        ax_e, _, _ = self.canvas_mapping["widget_E"]
        ax_e.clear()
        ax_e.set_ylabel('总能耗 (kWh)', fontsize=10)

    def create_canvas_with_toolbar(self, widget):
        """创建带工具栏的Matplotlib画布"""
        if widget.layout():
            while widget.layout().count():
                widget.layout().takeAt(0).widget().deleteLater()
        else:
            widget.setLayout(QVBoxLayout())

        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, widget)

        widget.layout().addWidget(toolbar)
        widget.layout().addWidget(canvas)
        return ax, canvas, toolbar

    def select_file(self, data_type):
        """文件选择对话框"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择CSV文件", "", "CSV Files (*.csv)")

        if file_path:
            if data_type == 'base':
                self.lineEdit_base.setText(file_path)
            else:
                self.lineEdit_opt.setText(file_path)
            self.auto_load_check()

    def auto_load_check(self):
        """自动加载并验证数据"""
        base_path = self.lineEdit_base.text()
        opt_path = self.lineEdit_opt.text()

        if base_path and opt_path:
            try:
                # 加载并排序数据
                self.df_fixed = pd.read_csv(base_path).sort_values('Timestamp')
                self.df_ga = pd.read_csv(opt_path).sort_values('Timestamp')

                # 验证必要字段
                required_columns = ['Timestamp', 'COP', 'Total_Power']
                for df in [self.df_fixed, self.df_ga]:
                    if not all(col in df.columns for col in required_columns):
                        raise ValueError("CSV文件缺少必要列")

                # 更新界面
                self.update_initial_plots()
                self.update_comparison_plots()
                self.generate_report()

            except Exception as e:
                QMessageBox.critical(self, "错误", f"数据加载失败：\n{str(e)}")
                logging.error(f"数据加载错误：{str(e)}")
                self.df_fixed = None
                self.df_ga = None

    def setup_combobox_connections(self):
        """绑定下拉菜单事件"""
        self.comboBox_1 = self.findChild(QComboBox, "comboBox_1")
        self.comboBox_2 = self.findChild(QComboBox, "comboBox_2")
        self.comboBox_3 = self.findChild(QComboBox, "comboBox_3")
        self.comboBox_4 = self.findChild(QComboBox, "comboBox_4")

        self.comboBox_1.currentTextChanged.connect(
            lambda: self.update_plot("widget_base1", self.comboBox_1, False))
        self.comboBox_2.currentTextChanged.connect(
            lambda: self.update_plot("widget_base2", self.comboBox_2, False))
        self.comboBox_3.currentTextChanged.connect(
            lambda: self.update_plot("widget_opt1", self.comboBox_3, True))
        self.comboBox_4.currentTextChanged.connect(
            lambda: self.update_plot("widget_opt2", self.comboBox_4, True))

    def update_initial_plots(self):
        """更新所有监控图表"""
        if self.df_fixed is not None and self.df_ga is not None:
            self.update_plot("widget_base1", self.comboBox_1, False)
            self.update_plot("widget_base2", self.comboBox_2, False)
            self.update_plot("widget_opt1", self.comboBox_3, True)
            self.update_plot("widget_opt2", self.comboBox_4, True)

    def update_plot(self, widget_name, combobox, is_opt):
        """更新单个监控图表"""
        if self.df_fixed is None or self.df_ga is None:
            return

        selected_text = combobox.currentText()
        mapping = self.option_mapping_opt if is_opt else self.option_mapping
        column = mapping.get(selected_text)
        df = self.df_ga if is_opt else self.df_fixed

        if not column or column not in df.columns:
            logging.error(f"无效的选项映射: {selected_text}")
            return

        ax, canvas, _ = self.canvas_mapping[widget_name]
        ax.clear()

        # 绘制数据曲线
        ax.plot(df['Timestamp'], df[column],
                color=colors[list(mapping.keys()).index(selected_text) % len(colors)],
                label=selected_text.split('_')[0])

        # 设置图表属性
        ax.set_title(selected_text.split('_')[0], fontsize=12)
        ax.set_xlabel('时间 (小时)', fontsize=10)
        ax.set_ylabel(self.get_unit(selected_text), fontsize=10)
        ax.xaxis.set_major_locator(MultipleLocator(30))
        ax.grid(True)
        ax.legend()

        # 更新画布
        canvas.figure.tight_layout()
        canvas.draw_idle()

    def get_unit(self, text):
        """获取单位标签"""
        units = {
            "流量": "(m³/h)",
            "温度": "(℃)",
            "COP": "",
            "功率": "(kW)"
        }
        for key in units:
            if key in text:
                return units[key]
        return ""

    def update_comparison_plots(self):
        """更新对比分析图表"""
        if self.df_fixed is None or self.df_ga is None:
            return

        # 更新COP对比曲线
        ax_cop, canvas_cop, _ = self.canvas_mapping["widget_COP"]
        ax_cop.clear()
        ax_cop.plot(self.df_ga['Timestamp'], self.df_ga['COP'],
                    label='优化策略', color=colors[0])
        ax_cop.plot(self.df_fixed['Timestamp'], self.df_fixed['COP'],
                    label='基准策略', linestyle='--', color=colors[1])
        ax_cop.legend()
        canvas_cop.draw_idle()

        # 更新能耗对比柱状图
        ax_e, canvas_e, _ = self.canvas_mapping["widget_E"]
        ax_e.clear()
        total_energy = [
            self.df_ga['Total_Power'].sum() * (1 / 60),  # 假设每分钟一个数据点
            self.df_fixed['Total_Power'].sum() * (1 / 60)
        ]
        sns.barplot(x=['优化策略', '基准策略'], y=total_energy,
                    palette=[colors[0], colors[1]], ax=ax_e)
        canvas_e.draw_idle()

    def calculate_metrics(self):
        """计算关键指标（带除以零保护）"""
        # 使用正确的变量名：df_ga（优化数据）和 df_fixed（基准数据）
        self.avg_cop_optimized = self.df_ga['COP'].mean()
        self.avg_cop_baseline = self.df_fixed['COP'].mean()

        if self.avg_cop_baseline != 0:
            self.cop_improvement = (self.avg_cop_optimized - self.avg_cop_baseline) / self.avg_cop_baseline * 100
        else:
            self.cop_improvement = 0

        # 动态计算时间间隔（与Ep.py保持一致）
        time_diff = (self.df_ga['Timestamp'].iloc[1] - self.df_ga['Timestamp'].iloc[0]) / 60  # 转换为小时
        self.total_energy_optimized = self.df_ga['Total_Power'].sum() * time_diff
        self.total_energy_baseline = self.df_fixed['Total_Power'].sum() * time_diff
        self.energy_saving = self.total_energy_baseline - self.total_energy_optimized
        self.energy_improvement = self.energy_saving / self.total_energy_baseline * 100 if self.total_energy_baseline != 0 else 0

    def generate_report(self):
        """生成分析报告"""
        self.calculate_metrics()
        report = f"""EP协同仿真报告
    ========================
    性能指标对比：
    ----------------
    1. COP对比：
       - 优化策略平均COP: {self.avg_cop_optimized:.2f}
       - 基准策略平均COP: {self.avg_cop_baseline:.2f}
       - COP提升百分比: {self.cop_improvement:.2f}%

    2. 能耗对比：
       - 优化策略总能耗: {self.total_energy_optimized:.2f} kWh
       - 基准策略总能耗: {self.total_energy_baseline:.2f} kWh
       - 节能量: {self.energy_saving:.2f} kWh
       - 节能百分比: {self.energy_improvement:.2f}%
    """
        self.textEdit.setPlainText(report)

    def export_report(self):
        """导出报告"""
        options = QFileDialog.Option.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(
            self, "保存报告", "", "文本文件 (*.txt);;所有文件 (*)", options=options)
        if file_name:
            try:
                with open(file_name, 'w', encoding='utf-8') as f:
                    f.write(self.textEdit.toPlainText())
                QMessageBox.information(self, "成功", f"报告已保存至:\n{file_name}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败:\n{str(e)}")

    def btn_page1(self):
        """切换监控页面"""
        self.btn_base_sim = self.findChild(QPushButton, "btn_base_sim")
        self.btn_opt_sim = self.findChild(QPushButton, "btn_opt_sim")
        self.btn_base_sim.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        self.btn_opt_sim.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))

    def btn_page2(self):
        """切换分析页面"""
        self.pushButton_COP = self.findChild(QPushButton, "pushButton_COP")
        self.pushButton_E = self.findChild(QPushButton, "pushButton_E")
        self.pushButton_COP.clicked.connect(lambda: self.stacked_widget_2.setCurrentIndex(0))
        self.pushButton_E.clicked.connect(lambda: self.stacked_widget_2.setCurrentIndex(1))


if __name__ == "__main__":
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec()