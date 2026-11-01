# 로그인 GUI # index0
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFrame, QMessageBox
import sys
sys.path.append('src')
from database import DatabaseManager

class LoginForm(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.db = DatabaseManager("data/database.db")
        self.face_auth_success = False # True / False
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(40, 50, 40, 50)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 얼굴 인식 확인 버튼
        self.face_auth_btn = QPushButton("🐵")
        self.face_auth_btn.setObjectName("FaceAuthBTN")
        self.face_auth_btn.clicked.connect(self.handle_face_auth)

        self.id_textedit = QLabel("🐵을(를) 눌러 얼굴 인식을 완료해주세요.")
        self.id_textedit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.id_textedit.setStyleSheet("""
            QLabel {
                background-color: ##F5F5F5;
                padding: 10px 15px;
                font-size: 13px;
                color: #666666;
            }
        """)

        self.pw_textedit = QLineEdit()
        self.pw_textedit.setPlaceholderText("비밀번호를 입력해주세요.")
        self.pw_textedit.setEchoMode(QLineEdit.EchoMode.Password)

        self.login_btn = QPushButton("로그인")
        self.login_btn.setObjectName("LoginBTN")
        self.login_btn.clicked.connect(self.handle_login)

        self.line = QFrame()
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setStyleSheet("QFrame { color: #D0D0D0; }")

        self.register_btn = QPushButton("유저 등록")
        self.register_btn.setObjectName("RegisterBTN")
        self.register_btn.clicked.connect(self.handle_register)

        layout.addWidget(self.face_auth_btn, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(20)
        layout.addWidget(self.id_textedit)
        layout.addWidget(self.pw_textedit)
        layout.addSpacing(10)
        layout.addWidget(self.login_btn)
        layout.addSpacing(10)
        layout.addWidget(self.line)
        layout.addSpacing(10)
        layout.addWidget(self.register_btn)

        self.setLayout(layout)

    def handle_face_auth(self):
        self.controller.switch_to_screen(1)

    def mark_face_auth_success(self):
        if self.controller.current_user is not None:
            username = self.db.get_username_by_id(self.controller.current_user)
        self.face_auth_success = True
        self.id_textedit.setText(f"안녕하세요, {username}님.\n비밃번호를 입력해 로그인을 완료해주세요.")
        self.face_auth_btn.setStyleSheet("background-color: #A6FF4D;")

    def handle_login(self):
        password = self.pw_textedit.text()
        
        if not password:
            QMessageBox.warning(self, "알림", "비밀번호를 입력해주세요.")
            return
        
        # 데이터베이스에서 사용자 확인
        success = self.db.check_user(self.controller.current_user, password)

        if self.face_auth_success:        
            if success:
                QMessageBox.information(self, "로그인 성공", "비밀번호 인증이 완료되었습니다.")
                # 입력 필드 초기화
                self.id_textedit.clear()
                self.pw_textedit.clear()
                # 메인 화면으로 이동
                self.controller.switch_to_screen(2)
            else:
                QMessageBox.warning(self, "오류", "틀린 비밀번호입니다.")
        else:
            QMessageBox.warning(self, "오류", "얼굴 인식을 먼저 완료해주세요.")

    def handle_register(self):
        self.controller.switch_to_screen(3)