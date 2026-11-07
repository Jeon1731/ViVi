import cv2
import sys
from torchvision import transforms
import torch
import torch.nn.functional as F
import numpy as np

# test1) Face Detection
yunet_model = "tests/assets/face_detection_yunet_2023mar.onnx"

img = cv2.imread('tests/assets/test2.jpg')
if img is None:
    sys.exit('파일이 없습니다.')

h, w = img.shape[:2]

# YuNet Detector
detector = cv2.FaceDetectorYN.create(yunet_model, "", input_size=(w, h), score_threshold=0.6, nms_threshold=0.3, top_k=5000)
detector.setInputSize((w, h))
retval, faces = detector.detect(img)

best_confidence = 0
best_face = -1
if faces is not None and len(faces) > 0:
    for i, face in enumerate(faces):
        if face[14] > best_confidence:
            best_confidence = face[14]
            best_face = i

face = faces[best_face]
print(face)
ulx, uly, boxw, boxh = face[0:4]

if face is not None:
    cv2.rectangle(img, (int(ulx), int(uly)), (int(ulx+boxw), int(uly+boxh)), (255, 0, 0), 3)

# cv2.imshow('Face Detection', img)

# test2) Face Recognition
vivi_model = "tests/assets/vivi_torchscript.pt"

img_2 = cv2.imread('tests/assets/test2.jpg')
if img_2 is None:
    sys.exit('파일이 없습니다.')

crop_img = img_2[int(uly):int(uly+boxh), int(ulx):int(ulx+boxw)]
resize_img = cv2.resize(crop_img, (224, 224), fx=0, fy=0)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.562, 0.433, 0.386],
        std=[0.325, 0.280, 0.258],
    ),
])

inp_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
inp_tensor = preprocess(inp_img).unsqueeze(0).to("cpu")

model = torch.jit.load(vivi_model, map_location="cpu")
model.eval()

with torch.inference_mode():
    classification_logits, embedding = model(inp_tensor)
    probabilities = classification_logits.softmax(dim=1)[0].cpu()
    embedding = F.normalize(embedding, p=1, dim=1)[0].cpu()

result = None    
if probabilities[1] > probabilities[0]:
    result = "attack1\nprint"
elif probabilities[2] > probabilities[0]:
    result = "attack2\nmask"
elif probabilities[3] > probabilities[0]:
    result = "attack3\nreplay(phone)"
elif probabilities[4] > probabilities[0]:
    result = "attack\nreplay(tablet)"
else: # real
    result = "real"
print(result)

img = cv2.resize(img, (224, 224), fx=0, fy=0)
combined = np.hstack((img, resize_img))
cv2.putText(combined, "Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.putText(combined, result, (224 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imshow("mannequin result", combined)
cv2.imwrite("tests/assets/test2_result.jpg", combined)
cv2.waitKey(0)
