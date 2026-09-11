import threading

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from database import DatabaseManager
from vivi.FaceDetector import FaceDetector
from vivi.FaceRecognizer import FaceRecognizer


class MonitorProtector(QThread):
    check_finished = Signal(bool)
    error = Signal(str)

    COSINE_DISTANCE_THRESHOLD = 0.637

    def __init__(self, username, time_window=3, parent=None):
        super().__init__(parent)
        self.username = username
        self.time_window = time_window
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()
        if self.isRunning() and threading.current_thread() is not self:
            self.wait()

    def run(self):
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            self.error.emit("카메라를 열 수 없습니다.")
            return

        database = DatabaseManager("data/users.db")
        registered_vector = database.get_user_vector(self.username)
        if registered_vector is None:
            self.error.emit("현재 사용자의 등록된 얼굴 정보를 찾을 수 없습니다.")
            capture.release()
            return

        try:
            while not self._stop_event.is_set():
                is_same_user = self._check_current_frame(capture, registered_vector)
                self.check_finished.emit(bool(is_same_user))
                self._stop_event.wait(self.time_window)
        except Exception as exception:
            self.error.emit(f"얼굴 확인 중 오류가 발생했습니다: {exception}")
        finally:
            capture.release()

    def _check_current_frame(self, capture, registered_vector):
        captured, frame = capture.read()
        if not captured:
            return False

        detector = FaceDetector(max_frames=1)
        detector.run(frame)
        if detector.best_frame is None or detector.best_face[14] <= 0:
            return False

        recognizer = FaceRecognizer()
        recognizer.run(detector.best_frame, detector.best_face)
        if recognizer.face_vector is None:
            return False

        current = np.asarray(recognizer.face_vector, dtype=np.float32).reshape(-1)
        registered = np.asarray(registered_vector, dtype=np.float32).reshape(-1)
        if current.size != registered.size:
            return False

        current_norm = np.linalg.norm(current)
        registered_norm = np.linalg.norm(registered)
        if current_norm == 0 or registered_norm == 0:
            return False

        cosine_distance = 1.0 - np.dot(current, registered) / (
            current_norm * registered_norm
        )
        return bool(cosine_distance <= self.COSINE_DISTANCE_THRESHOLD)