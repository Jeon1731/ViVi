# 메인 화면 # index2
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton
import sys
sys.path.append('src')
from database import DatabaseManager
from datetime import datetime
from vivi.MonitorProtector import MonitorProtector

class MainForm(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.db = DatabaseManager("data/users.db")
        self.monitor_protector = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 40, 30 ,40)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.avatar = QLabel("🐵")
        self.avatar.setFixedSize(130, 130)
        self.avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.avatar.setFont(QFont("Arial", 42))
        self.avatar.setStyleSheet("background-color: #E0E0E0; border-radius: 65px;")

        self.toggle_btn = QPushButton("▶️")
        self.toggle_btn.setFixedSize(50, 50)
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #A6FF4D;
                border-radius: 25px;
                font-size: 18px;
            }
            QPushButton:hover { background-color: #95E644; }
        """)
        self.toggle_btn.clicked.connect(self.handle_background_process)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.load_logs()
        self.log_display.setStyleSheet("""
            QTextEdit {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
                color: #333333;
                padding: 10px;
            }
        """)

        layout.addWidget(self.avatar, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.toggle_btn, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.log_display)

        self.setLayout(layout)

    def handle_background_process(self):
        if self.toggle_btn.isChecked():
            self.toggle_btn.setText("⏸️")
            self.start_monitoring()
        else:
            self.toggle_btn.setText("▶️")
            self.stop_monitoring()

    def start_monitoring(self):
        self.stop_monitoring()
        self.monitor_protector = MonitorProtector(self.controller.current_user, 5, self)
        self.monitor_protector.check_finished.connect(self.handle_monitor_result)
        self.monitor_protector.error.connect(self.handle_monitor_error)
        self.monitor_protector.start()

    def stop_monitoring(self):
        if self.monitor_protector and self.monitor_protector.isRunning():
            self.monitor_protector.stop()
        self.monitor_protector = None

    def handle_monitor_result(self, is_verified):
        if is_verified:
            if self.controller.central_stacked.currentIndex() == 5:
                self.controller.switch_to_screen(2)
        elif self.controller.central_stacked.currentIndex() != 5:
            self.controller.switch_to_screen(5)

    def handle_monitor_error(self, message):
        self.stop_monitoring()
        self.toggle_btn.setChecked(False)
        self.toggle_btn.setText("▶️")
        self.log_display.setPlainText(message)

    def load_logs(self):
        """현재 사용자 로그 조회 및 표시"""
        if self.controller.current_user:
            logs = self.db.get_logs(self.controller.current_user)
            log_text = self.format_logs(logs)
            self.log_display.setPlainText(log_text)
        else:
            self.log_display.setPlainText("로그인 후 조회됩니다.")

    def format_logs(self, logs):
        """로그 포맷팅"""
        if not logs:
            return "아직 기록된 로그가 없습니다."
        
        formatted_logs = []
        for log_id, username, log_type, date_time in logs:
            # date 형식: "2026-08-17 10:30:45"
            try:
                log_time = datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S")
                time_str = log_time.strftime("%H:%M")
            except:
                time_str = date_time[:5]  # HH:MM 부분만 추출
            
            formatted_logs.append(f"{time_str} | {log_type}")
        
        return "\n".join(formatted_logs)

    def update_logs(self):
        """로그 새로고침"""
        self.load_logs()

    def close_monitoring(self):
        self.stop_monitoring()