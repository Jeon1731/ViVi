import cv2
import numpy as np

def adjust_gamma(image, gamma=1.0):
    """감마 보정 함수 (LUT 방식)"""
    invGamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)
    ]).astype("uint8")
    return cv2.LUT(image, table)

def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    """컬러 이미지 대응 CLAHE 적용 함수"""
    # 1. BGR 이미지를 YCrCb 색 공간으로 변환 (밝기 채널 분리용)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    # 2. CLAHE 객체 생성 및 밝기(Y) 채널에 적용
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    y_enhanced = clahe.apply(y)
    
    # 3. 채널 다시 병합 및 BGR 변환
    merged = cv2.merge([y_enhanced, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

# 이미지 읽기
img = cv2.imread("tests/assets/sample4.jpg")
img = cv2.resize(img, dsize=(0, 0), fx=0.25, fy=0.25)

if img is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인해주세요.")
else:
    # 1. 감마 보정 적용 (1.5로 설정하여 어두운 곳을 밝게)
    gamma_img = adjust_gamma(img, gamma=1.5)

    # 2. CLAHE 적용
    clahe_img = apply_clahe(img, clip_limit=2.0, grid_size=(8, 8))

    # 3. 한눈에 비교하기 위해 가로로 병합 (가로 해상도가 너무 커지면 각각 imshow 하셔도 됩니다)
    combined = np.hstack((img, gamma_img, clahe_img))

    # 이미지 위에 텍스트 라벨 추가 (선택 사항)
    h, w, _ = img.shape
    cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(combined, "Gamma (1.5)", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(combined, "CLAHE", (w * 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 결과창 출력
    cv2.imshow("Comparison: Original vs Gamma vs CLAHE", combined)
    cv2.imwrite("tests/assets/sample4_result.jpg", combined)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()