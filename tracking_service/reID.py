import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ReIDManager:
    def __init__(self, threshold=0.8):
        self.gallery = {}  # {global_id: embedding}
        self.threshold = threshold
        self.global_id_counter = 0

    def extract_embedding(self, img_np):
        try:
            resized = cv2.resize(img_np, (64, 128))
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                                [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist.reshape(1, -1)
        except Exception as e:
            print(f"Error extracting histogram: {e}")
            return None

    def match_or_create_global_id(self, embedding):
        if embedding is None:
            return None

        best_id = None
        best_score = -1

        for global_id, stored_emb in self.gallery.items():
            score = cosine_similarity(embedding, stored_emb)[0][0]
            if score > best_score:
                best_score = score
                best_id = global_id

        if best_score >= self.threshold:
            return best_id

        # Nếu không đủ điểm thì tạo ID mới
        self.global_id_counter += 1
        new_id = str(self.global_id_counter)
        self.gallery[new_id] = embedding
        return new_id
