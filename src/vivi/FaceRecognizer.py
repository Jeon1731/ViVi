import cv2

import vivi.ViVi as ViVi

class FaceRecognizer:
    def __init__(self):
        """ViVI(MobileFaceNetV3) Recognizer"""
        self.vivi_model = "src/vivi/vivi_torchscript.pt"
        self.recognizer = None # (Temp) Singleton
        self.face_vector = None
        self.attack = None # 0: real, 1: attack

    def run(self, frame, face):
        self.recognizer = ViVi.FaceRecognizerMFNV3.create(self.vivi_model) # model
        aligned_face = self.recognizer.alignCrop(frame, face)
        self.attack, self.face_vector = self.recognizer.feature(aligned_face)