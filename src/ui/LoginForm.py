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
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(40, 50, 40, 50)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 고정 더미 아바타 (로그인 전)
        self.avatar = QLabel("🐵")
        self.avatar.setFixedSize(150, 150)
        self.avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.avatar.setFont(QFont("Arial", 48))
        self.avatar.setStyleSheet("background-color: #E0E0E0; border-radius: 75px;")

        self.id_textedit = QLineEdit()
        self.id_textedit.setPlaceholderText("아이디를 입력해주세요.")

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

        layout.addWidget(self.avatar, 0, Qt.AlignmentFlag.AlignCenter)
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

    def handle_login(self):
        username = self.id_textedit.text().strip()
        password = self.pw_textedit.text()
        
        if not username or not password:
            QMessageBox.warning(self, "알림", "아이디와 비밀번호를 입력해주세요.")
            return
        
        # 데이터베이스에서 사용자 확인
        success, message, user_id = self.db.check_user(username, password)
        
        if success:
            QMessageBox.information(self, "성공", message)
            # 현재 사용자 저장
            self.controller.current_user = user_id
            # 입력 필드 초기화
            self.id_textedit.clear()
            self.pw_textedit.clear()
            # 얼굴 인증 화면으로 이동
            self.controller.switch_to_screen(1)
        else:
            QMessageBox.warning(self, "오류", message)

    def handle_register(self):
        self.controller.switch_to_screen(3)