# main.py
import sys
import os
from PyQt5 import QtWidgets
from main_window import MainWindow






### numpy 1.26.4 原本是numpy                    2.2.6
### opencv-python 4.12.0.88
### h5py 3.11.0
### scipy                  1.15.3









def fix_qt_plugin_issues():
    """在创建QApplication之前修复Qt插件问题"""
    # 清除所有可能冲突的环境变量
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    os.environ.pop("QT_PLUGIN_PATH", None)
    
    # 设置正确的平台
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    
    # 添加系统Qt插件路径
    possible_plugin_paths = [
        "/usr/lib/x86_64-linux-gnu/qt5/plugins",
        "/usr/lib/qt/plugins",
        "/usr/local/lib/qt/plugins",
        os.path.join(sys.prefix, "plugins"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "qt_plugins")
    ]
    
    # 查找有效的插件路径
    valid_plugin_path = None
    for path in possible_plugin_paths:
        xcb_plugin = os.path.join(path, "platforms", "libqxcb.so")
        if os.path.exists(xcb_plugin):
            valid_plugin_path = path
            break
    
    if valid_plugin_path:
        os.environ["QT_PLUGIN_PATH"] = valid_plugin_path
    else:
        print("警告: 未找到有效的Qt平台插件路径")
        
    # 调试信息
    print(f"设置 QT_PLUGIN_PATH = {os.environ.get('QT_PLUGIN_PATH', '未设置')}")
    print(f"设置 QT_QPA_PLATFORM = {os.environ.get('QT_QPA_PLATFORM', '未设置')}")

if __name__ == "__main__":
    # 先修复Qt插件问题
    fix_qt_plugin_issues()
    
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    
    mainWindow = MainWindow()
    mainWindow.setWindowTitle("PIV - 粒子图像测速系统 (事件相机版)")
    mainWindow.show()
    sys.exit(app.exec_())