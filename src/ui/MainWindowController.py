# 메인 윈도우 컨트롤러 (화면 흐름 제어)
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QStackedWidget

from LoginForm import LoginForm
from FaceAuthForm import FaceAuthForm
from MainForm import MainForm
from RegistrationForm import RegistrationForm
from FaceRegistForm import FaceRegistForm
from TopLockForm import TopLockForm

class MainWindowContoller(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ViVi")
        self.setFixedSize(380, 620)
        self.normal_window_flags = self.windowFlags()
        self.current_user = None  # 현재 로그인한 사용자

        self.central_stacked = QStackedWidget()
        self.setCentralWidget(self.central_stacked)

        self.login_form = LoginForm(self)
        self.face_auth_form = FaceAuthForm(self)
        self.main_form = MainForm(self)
        self.registration_form = RegistrationForm(self)
        self.face_regist_form = FaceRegistForm(self)
        self.registration_form.face_regist_form = self.face_regist_form
        self.face_regist_form.face_registed.connect(self.registration_form.handle_face_registered)
        self.top_lock_form = TopLockForm(self)

        self.central_stacked.addWidget(self.login_form)         #index 0
        self.central_stacked.addWidget(self.face_auth_form)     #index 1
        self.central_stacked.addWidget(self.main_form)          #index 2
        self.central_stacked.addWidget(self.registration_form)  #index 3
        self.central_stacked.addWidget(self.face_regist_form)   #index 4
        self.central_stacked.addWidget(self.top_lock_form)      #index 5
        
        
        # 전역 스타일시트
        self.setStyleSheet("""
            QWidget {
                background-color: #F5F5F5;
                color: #333333;
                font-family: 'Malgun Gothic', Arial, sans-serif;
            }
            QLineEdit {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 20px;
                padding: 10px 15px;
                font-size: 13px;
                color: #666666;
            }
            QLineEdit:focus { border: 1px solid #A6FF4D; }
            QPushButton#LoginBTN {
                background-color: #A6FF4D;
                border: none;
                border-radius: 20px;
                padding: 12px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton#LoginBTN:hover { background-color: #95E644; }
            QPushButton#RegisterBTN {
                background-color: #FFFFFF;
                border: 1px solid #D0D0D0;
                border-radius: 20px;
                padding: 12px;
                font-size: 13px;
                color: #555555;
                }
            QPushButton#RegisterBTN:hover { background-color: #EEEEEE; }
            QPushButton#FaceRegisterBTN {
                background-color: #FFFFFF;
                border: 1px solid #D0D0D0;
                border-radius: 20px;
                padding: 12px;
                font-size: 13px;
                color: #555555;
            }
            QPushButton#FaceRegisterBTN:hover { background-color: #EEEEEE; }
            QPushButton#UserRegisterBTN {
                background-color: #FFFFFF;
                border: 1px solid #D0D0D0;
                border-radius: 20px;
                padding: 12px;
                font-size: 13px;
                color: #555555;
            }
            QPushButton#UserRegisterBTN:hover { background-color: #EEEEEE; }
            """)

    def switch_to_screen(self, index):
        self.central_stacked.setCurrentIndex(index)
        if index == 5:
            self.setWindowFlags(
                self.normal_window_flags
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.FramelessWindowHint
            )
            self.showFullScreen()
            self.top_lock_form.start_lock()
        else:
            self.top_lock_form.stop_lock()
            self.setWindowState(
                self.windowState() & ~Qt.WindowState.WindowFullScreen
            )
            self.setWindowFlags(self.normal_window_flags)
            self.showNormal()
            self.setFixedSize(380, 620)
            self.show()
        # Face Auth Form으로 진입할 때 카메라 가동 시작
        if index == 1: #FaceAuthForm
            self.face_auth_form.start_camera()
        if index == 2: #MainForm
            self.main_form.update_logs()
        if index == 4: #FaceRegistForm
            self.face_regist_form.start_camera()

    def closeEvent(self, event):
        self.main_form.close_monitoring()
        # 창이 닫힐 때 카메라 스레드가 켜져있다면 확실하게 종료
        if hasattr(self.face_auth_form, 'camera_thread') and self.face_auth_form.camera_thread:
            self.face_auth_form.camera_thread.stop()
            event.accept()
