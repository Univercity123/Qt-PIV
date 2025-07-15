import sys
import os
from PyQt5 import QtWidgets
from main_window import MainWindow

# 解决 Qt 插件问题
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""
os.environ["QT_QPA_PLATFORM"] = "xcb"

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    mainWindow = MainWindow()
    mainWindow.setWindowTitle("PIV - 粒子图像测速系统")
    mainWindow.show()
    sys.exit(app.exec_())