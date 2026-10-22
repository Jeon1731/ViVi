# 얼굴 인식 화면 # index1
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QMessageBox
import cv2
import numpy as np

import sys
sys.path.append('src')
from database import DatabaseManager

from CircularCameraWidget import CircularCameraWidget
from CameraThread import CameraThread

from vivi.FaceDetector import FaceDetector
from vivi.FaceRecognizer import FaceRecognizer

class FaceAuthForm(QWidget):
    face_authorized = Signal(object)
    COSINE_DISTANCE_THRESHOLD = 0.637

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.db = DatabaseManager("data/users.db")
        self.camera_thread = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(25)
        layout.setContentsMargins(40, 50, 40, 50)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.title_label = QLabel("얼굴을 인식중입니다...")
        self.title_label.setFont(QFont("Malgun Gothic", 16, QFont.Weight.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 실제 웹캠 영상이 출력될 원형 위젯
        self.camera_view = CircularCameraWidget(300)

        self.sub_label = QLabel("정면을 바라봐주세요")
        self.sub_label.setFont(QFont("Malgun Gothic", 12))
        self.sub_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.title_label)
        layout.addWidget(self.camera_view, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.sub_label)

        self.setLayout(layout)

    def start_camera(self):
        # 백그라운드 스레드로 카메라 작동 시동
        self.camera_thread = CameraThread()

        # Face Detector 생성
        self.face_detector = FaceDetector()
        self.camera_thread.frame_received.connect(self.face_detector.run)
        self.face_detector.frame_limit_reached.connect(self.finish_face_authorization)

        self.camera_thread.qimage_frame_received.connect(self.camera_view.update_frame)
        self.camera_thread.start()

    def finish_face_authorization(self, best_frame, best_face):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()

        if best_frame is None or best_face[14] <= 0:
            QMessageBox.warning(self, "오류", "얼굴을 찾지 못했습니다. 다시 시도해주세요.")
            self.controller.switch_to_screen(0)
            return

        self.face_recognizer = FaceRecognizer()
        self.face_recognizer.run(best_frame, best_face)

        if self.face_recognizer.face_vector is not None and self.face_recognizer.attack == 0:
            stored_vector = self.db.get_user_vector(self.controller.current_user)
            if stored_vector is None: #(Temp)
                QMessageBox.warning(self, "오류", "등록된 얼굴 정보를 찾을 수 없습니다.")
                self.controller.switch_to_screen(0)
                return

            current_vector = np.asarray(self.face_recognizer.face_vector, dtype=np.float32)
            registered_vector = np.asarray(stored_vector, dtype=np.float32)
            if current_vector.size != registered_vector.size: #(Temp)
                QMessageBox.warning(self, "오류", "얼굴 벡터 형식이 올바르지 않습니다.")
                self.controller.switch_to_screen(0)
                return

            current_norm = np.linalg.norm(current_vector)
            registered_norm = np.linalg.norm(registered_vector)
            if current_norm == 0 or registered_norm == 0: #(Temp)
                QMessageBox.warning(self, "오류", "얼굴 벡터가 올바르지 않습니다.")
                self.controller.switch_to_screen(0)
                return

            cosine_similarity = np.dot(current_vector, registered_vector) / (
                current_norm * registered_norm
            )
            cosine_distance = 1.0 - cosine_similarity
            is_same_user = cosine_distance <= self.COSINE_DISTANCE_THRESHOLD

            if is_same_user:
                self.face_authorized.emit(self.controller.current_user)
                self.controller.switch_to_screen(2)
            else:
                QMessageBox.warning(self, "인증 실패", "등록된 사용자와 일치하지 않습니다.")
                self.controller.switch_to_screen(0)
        elif self.face_recognizer.face_vector is not None and self.face_recognizer == 1:
            QMessageBox.warning(self, "경고", "실제 얼굴을 인식해주세요.")
            self.controller.switch_to_screen(0)
        else:
            QMessageBox.warning(self, "오류", "얼굴 인식에 실패하였습니다.")
            self.controller.switch_to_screen(0)

    def stop_camera_and_proceed(self):
        if self.camera_thread and self.camera_thread.isRunning:
            self.camera_thread.stop()
        self.controller.switch_to_screen(2)