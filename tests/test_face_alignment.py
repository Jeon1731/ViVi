import cv2
import numpy as np
import sys
import math

yunet_model = "tests/assets/face_detection_yunet_2023mar.onnx"

img = cv2.imread('tests/assets/sample3.jpg')
if img is None:
    sys.exit('파일이 없습니다.')

img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)

h, w = img.shape[:2]

# YuNet detector
detector = cv2.FaceDetectorYN.create(yunet_model, "", input_size=(w, h), score_threshold=0.6, nms_threshold=0.3, top_k=5000)
detector.setInputSize((w, h))
retval, faces = detector.detect(img)

best_confidence = 0
best_face = -1
if faces is not None and len(faces) > 0:
    for i, face in enumerate (faces) :
        if face[14] > best_confidence:
            best_confidence = face[14]
            best_face = i

face = faces[best_face]
lex, ley, rex, rey, nx, ny = face[4:10]

left_eye = (lex, ley)
right_eye = (rex, rey)
nose = (nx, ny)

# cv2.circle(img, (int(lex), int(ley)), 3, (255, 0, 0), -1)
# cv2.circle(img, (int(rex), int(rey)), 3, (0, 255, 0), -1)
# cv2.circle(img, (int(nx), int(ny)), 3, (0, 0, 255), -1)

# cv2.imshow("check landmarks", img)
# cv2.waitKey(0)

# 1) point_3rd 계산
if left_eye[1] > right_eye[1]: # 반시계 방향으로 기울어짐
    point_3rd = (right_eye[0], left_eye[1])
    direction = -1 # 시계 방향 회전 필요
else: # 시계 방향으로 기울어짐
    point_3rd = (left_eye[0], right_eye[1])
    direction = 1 # 반시계 방향 회전 필요

# 2) 유클리드 길이(변의 길이) 구하기
dist_ab = math.dist(left_eye, right_eye)
dist_bc = math.dist(right_eye, point_3rd)

# 3) 각도 구하기
cos_a = dist_bc / dist_ab
angle = math.degrees(math.acos(cos_a))

# 4) 방향 구하기
if direction == -1:
    angle = 90 - angle

if point_3rd[0] == left_eye[0] and point_3rd[1] < left_eye[1]:
    angle = -angle

if point_3rd[0] == right_eye[0] and point_3rd[1] < right_eye[1]:
    angle = -angle

angle = direction * angle

# 5) Affine 변환 행렬 생성
M = cv2.getRotationMatrix2D(nose, angle, 1.0)

# 6) Affine 변환
aligned_img = cv2.warpAffine(img, M, (w, h))

# 7) 244x244 resize
box = face[0:4].astype(np.int32)
x, y, w, h = box
crop_img = aligned_img[y:y+h, x:x+w]

result = cv2.resize(crop_img, (244, 244), fx=0, fy=0)
cv2.imshow("result", result)
cv2.waitKey(0)