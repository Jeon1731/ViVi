import cv2
import sys
import numpy as np
import pickle

yunet_model = "tests/assets/face_detection_yunet_2023mar.onnx"
sface_model = "tests/assets/face_recognition_sface_2021dec.onnx"

imgname = 'sample2'
img = cv2.imread(f'tests/assets/{imgname}.jpg')    
if img is None : 
    sys.exit('파일이 없습니다.')

img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)

h, w = img.shape[:2]

# YuNet detector
detector = cv2.FaceDetectorYN.create(yunet_model, "", input_size=(w, h), score_threshold=0.6, nms_threshold=0.3, top_k=5000)
#(모델, 구성 파일(보통 ""), 입력크기, 결과 임계값, NMS  임계값, 최대검출수)
detector.setInputSize((w, h))           # 입력 크기 설정
retval, faces = detector.detect(img)    # 얼굴 검출

# 랜드마크 이름을 튜플로 정의 (SFace가 반환하는 5개의 점)
landmark_names = ("Right Eye", "Left Eye", "Nose", "Right Mouth Corner", "Left Mouth Corner")
landmark_colors = [(0, 255, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 0, 0)] # G, G, R, B, B

# SFace recognizer
recognizer = cv2.FaceRecognizerSF.create(sface_model, "")
   
if faces is not None and len(faces) > 0:
    embeddings = []
    for i, face in enumerate(faces):
        # Y)바운딩 박스 face[:4] = [x, y, w, h]
        x, y, fw, fh = face[0:4].astype(np.int32)
        cv2.rectangle(img, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

        # Y) 신뢰도 표시
        confidence = face[4]
        cv2.putText(img, f"ID{i}:{confidence:.2f}", (x,y-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)

        # Y) 랜드마크 5점 face[5:] = [4x, 5y, 6x, 7y, ..., l3x, l4y]  
        # 일부 OpenCV/DNN 환경에서는 랜드마크를 출력할 때 (Y, X) 순서로 저장하는 경우가 발생
        landmarks = face[5:15].astype(np.int32).reshape((5, 2))
        for j, (lm_y, lm_x) in enumerate(landmarks):
           cv2.circle(img, (lm_x, lm_y), 3, landmark_colors[j], -1) # 빨간 점으로 표시

        # S1) YuNet 결과를 기반으로 얼굴 정렬(aligned face) 추출
        aligned_face = recognizer.alignCrop(img, face)

        # S2) SFace로 128D 임베딩(특징 벡터) 추출
        feat = recognizer.feature(aligned_face)
        embeddings.append(feat)

    cv2.imshow("YuNet + SFace", img)
    cv2.imwrite(f"tests/assets/{imgname}_result.jpg", img)

    print(f"Detected {len(embeddings)} faces.")
    print("Embedding vector 예:", embeddings[0]) # 예: 두 얼굴 간 코사인 유사도 계산도 가능

    # 코사인 거리 비교용 임베딩 저장하기
    with open("tests/assets/face_embeddings.pkl", "ab") as f:
        pickle.dump(embeddings, f)

cv2.waitKey()
cv2.destroyAllWindows()