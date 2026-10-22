import cv2
from PySide6.QtCore import QObject, Signal, Slot

class FaceDetector(QObject):
    frame_limit_reached = Signal(object, object)

    def __init__(self, max_frames=60):
        super().__init__()
        """Prototype - YuNet Detector"""
        self.yunet_model = "src/vivi/face_detection_yunet_2023mar.onnx"
        self.detector = None # (Temp) Singleton
        self.best_frame = None # (Temp) 객체 안에 저장하지 않고 바로 전달될 수 있도록 변경
        self.best_face = [0 for _ in range(15)] # (Temp) 객체 안에 저장하지 않고 바로 전달될 수 있도록 변경
        self.max_frames = max_frames
        self.frame_count = 0

    @Slot(object)
    def run(self, frame):
        if self.frame_count >= self.max_frames:
            return

        h, w = frame.shape[:2]
        self.detector = cv2.FaceDetectorYN.create(self.yunet_model, "", input_size=(w, h), score_threshold=0.8)
        retval, faces = self.detector.detect(frame)
        self.frame_count += 1
        # 0~3: bbox (x, y, w, h)
        # 4~13: landmarks
        # 14: confidence score
        
        if faces is not None:
            for face in faces:
                confidence = face[14]
                if confidence > self.best_face[14]:
                    self.best_frame = frame.copy()
                    self.best_face = face

        if self.frame_count >= self.max_frames:
            self.frame_limit_reached.emit(self.best_frame, self.best_face)