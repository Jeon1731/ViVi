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
    unregistered_face_detected = Signal()

    COSINE_DISTANCE_THRESHOLD = 0.637
    UNREGISTERED_FACE_DETECTIONS_REQUIRED = 3

    def __init__(self, user_id, time_window=3, parent=None):
        super().__init__(parent)
        self.user_id = user_id
        self.time_window = time_window
        self._stop_event = threading.Event()
        self._unregistered_faces = []

    def stop(self):
        self._stop_event.set()
        if self.isRunning() and threading.current_thread() is not self:
            self.wait()

    def run(self):
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            self.error.emit("카메라를 열 수 없습니다.")
            return

        database = DatabaseManager("data/database.db")
        registered_vector = database.get_user_vector(self.user_id)
        if registered_vector is None:
            self.error.emit("현재 사용자의 등록된 얼굴 정보를 찾을 수 없습니다.")
            capture.release()
            return

        try:
            while not self._stop_event.is_set():
                is_same_user = self._check_current_frame(
                    capture, registered_vector, database
                )
                self.check_finished.emit(bool(is_same_user))
                self._stop_event.wait(self.time_window)
        except Exception as exception:
            self.error.emit(f"얼굴 확인 중 오류가 발생했습니다: {exception}")
        finally:
            capture.release()

    def _check_current_frame(self, capture, registered_vector, database):
        captured, frame = capture.read()
        if not captured:
            self._update_unregistered_faces([])
            return False

        detector = FaceDetector(max_frames=1)
        detector.run(frame)
        unknown_vectors = self._get_unregistered_face_vectors(
            frame, detector, database
        )
        self._update_unregistered_faces(unknown_vectors)

        if detector.best_frame is None or detector.best_face[14] <= 0:
            return False

        recognizer = FaceRecognizer()
        recognizer.run(detector.best_frame, detector.best_face)
        if recognizer.face_vector is None:
            return False

        if recognizer.attack != 0:
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

    def _get_unregistered_face_vectors(self, frame, detector, database):
        if detector.faces is None or detector.best_frame is None:
            return []

        unknown_vectors = []
        for face in detector.faces:
            if np.array_equal(face, detector.best_face):
                continue

            recognizer = FaceRecognizer()
            recognizer.run(frame, face)
            if recognizer.attack != 0 or recognizer.face_vector is None:
                continue
            if database.find_most_similar_user_id(recognizer.face_vector) is None:
                unknown_vectors.append(
                    np.asarray(recognizer.face_vector, dtype=np.float32).reshape(-1)
                )

        return unknown_vectors

    def _update_unregistered_faces(self, current_vectors):
        previous_faces = self._unregistered_faces
        next_faces = []
        matched_previous = set()

        for vector in current_vectors:
            matched_index = None
            for index, (previous_vector, _, _) in enumerate(previous_faces):
                if index in matched_previous:
                    continue
                distance = self._cosine_distance(vector, previous_vector)
                if distance <= self.COSINE_DISTANCE_THRESHOLD:
                    matched_index = index
                    break

            if matched_index is None:
                next_faces.append((vector, 1, False))
                continue

            matched_previous.add(matched_index)
            _, detected_count, already_alerted = previous_faces[matched_index]
            detected_count += 1
            if (
                detected_count >= self.UNREGISTERED_FACE_DETECTIONS_REQUIRED
                and not already_alerted
            ):
                self.unregistered_face_detected.emit()
                already_alerted = True
            next_faces.append((vector, detected_count, already_alerted))

        self._unregistered_faces = next_faces

    @staticmethod
    def _cosine_distance(first, second):
        first_norm = np.linalg.norm(first)
        second_norm = np.linalg.norm(second)
        if first_norm == 0 or second_norm == 0:
            return float("inf")
        return float(1.0 - np.dot(first, second) / (first_norm * second_norm))