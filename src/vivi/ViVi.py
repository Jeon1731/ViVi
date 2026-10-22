import torch
import math
import cv2
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

class FaceRecognizerMFNV3:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def create(cls, model):
        """
        map_location: GPU 학습 모델을 CPU 환경 등 다른 장치에서 불러올 때 위치 매핑
        weights_only: false(가중치+모델)
        """
        recognizer = cls()
        recognizer.model = torch.jit.load(model, map_location=recognizer.device)
        recognizer.model.eval()

        return recognizer

    def alignCrop(self, frame, face):
        # face
        # 0~3: bbox (x, y, w, h)
        # 4~13: landmarks
        # 14: confidence score
        """
        [원본 이미지 + 5개 랜드마크]
                ↓
        [두 눈의 각도 측정 및 기준점 확보]
                ↓
        [아핀 변환(회전·크기 조절) 행렬 계산]
                ↓
        [정면 정렬 및 표준 규격(224x224)으로 크롭] 
                ↓
        [최종 출력: 모델 입력용 정면 얼굴 이미지]   
        """
        h, w = frame.shape[:2]
        lex, ley, rex, rey, nx, ny = face[4:10]

        left_eye = (lex, ley)
        right_eye = (rex, rey)
        nose = (nx, ny)

        # 1) point_3rd 계산
        if left_eye[1] > right_eye[1]:
            point_3rd = (right_eye[0], left_eye[1])
            direction = -1
        else:
            point_3rd = (left_eye[0], right_eye[1])
            direction = 1

        # 2) 유클리드 길이 구하기
        dist_ab = math.dist(left_eye, right_eye)
        dist_bc = math.dist(right_eye, point_3rd)

        # 3) 각도 구하기
        cos_a = dist_bc / dist_ab
        angle = math.degrees(math.acos(cos_a))

        if direction == -1:
            angle = 90-angle

        # 4) 방향 구하기
        if point_3rd[0] == left_eye[0] and point_3rd[1] < left_eye[1]:
            angle = -angle

        if point_3rd[0] == right_eye[0] and point_3rd[1] < right_eye[1]:
            angle = -angle

        angle = direction * angle

        # 5) Affine 변환 행렬 생성
        M = cv2.getRotationMatrix2D(nose, angle, 1.0)

        # 6) Affine 변환
        aligned_img = cv2.warpAffine(frame, M, (w, h))

        # 7) 224x224 resize
        box = face[0:4].astype(np.int32)
        x, y, w, h = box
        crop_img = aligned_img[y:y+h, x:x+w]

        result = cv2.resize(crop_img, (224, 224), fx=0, fy=0)
        return result

    def feature(self, frame): # frame(224x224 aligned face)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.491, 0.372, 0.325],
                std=[0.364, 0.302, 0.269],
            ),
        ])

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess(frame).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            classification_logits, embedding = self.model(input_tensor)
            probabilities = classification_logits.softmax(dim=1)[0].cpu()
            embedding = F.normalize(embedding, p=2, dim=1)[0].cpu()

        if probabilities[0] > probabilities[1]: # real > attack
            return 0, embedding # retval, 128d embedding 
        else:
            return 1, embedding # retval, 128d embedding 
