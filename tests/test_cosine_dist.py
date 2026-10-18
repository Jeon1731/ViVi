import numpy as np
import pickle

def calculate_cosine_similarity(v1, v2):
    """두 벡터의 코사인 유사도 계산"""
    v1 = np.squeeze(v1)
    v2 = np.squeeze(v2)

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

def match_faces(first_embeddings, second_embeddings, threshold=0.6):
    """
    first_embeddings와 second_embbedings의 각 벡터를 비교하여
    유사한 얼굴이 있으면 해당 embedding의 index 번호를 출력합니다. 
    """
    for idx2, vec2 in enumerate(second_embeddings):
        best_match_idx = -1
        max_similarity = -1.0

        # first_embeddings에 저장된 임베딩들과 하나씩 비교
        for idx1, vec1 in enumerate(first_embeddings):
            similarity = calculate_cosine_similarity(vec2, vec1)

            # 가장 높은 유사도를 가진 인덱스를 찾음
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_idx = idx1

        # 가장 높은 유사도가 기준치(threshold)를 넘었는지 확인
        if max_similarity >= threshold:
            print(f"[sample2 ID: {idx2}] 유사한 얼굴 -> sample1 ID: {best_match_idx} (유사도: {max_similarity:.4f})")
        else:
            print(f"[sample2 ID: {idx2}] sample1과 유사한 얼굴이 없습니다.")

# 임베딩 불러오기
with open("tests/assets/face_embeddings.pkl", "rb") as f:
    first_embeddings = pickle.load(f)   #첫 번째로 저장했던 리스트
    second_embeddings = pickle.load(f)  #두 번째로 추가했던 리스트

match_faces(first_embeddings, second_embeddings, threshold=0.6)