import numpy as np
import insightface
import pandas as pd
import cv2
import os
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, Index

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

################### INITIAL MILVUS #####################
connections.connect("default", host="localhost", port="19530")
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
collection = Collection("face_embeddings")
# Release collection trước khi tạo index
collection.release()
index_params = {
    "metric_type": "L2",  # Sử dụng Euclidean distance, có thể thay bằng "COSINE" nếu dùng cosine similarity
    "index_type": "IVF_FLAT",  # Chọn loại index, ví dụ IVF_FLAT
    "params": {"nlist": 128}  # Tham số nlist, quy định số lượng cluster cho IVF
}
index = Index(collection, "embedding", index_params)
collection.load()
######################################################

################ LOAD INSIGHT MODLE ##################
model = insightface.app.FaceAnalysis(allowed_modules=['detection', 'recognition'])
model.prepare(ctx_id=0)  # -1 for CPU, use GPU ID if available (e.g., 0 for GPU)
######################################################

################ LOAD CAMERA ##################
video_path = 0
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: can't read video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video is final or can't read frame.")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = model.get(frame)
    if len(faces) > 0:
        for face in faces:
            bbox = face['bbox']
            bbox_int = [int(value) for value in bbox]
            x1 = bbox_int[0]
            y1 = bbox_int[1]
            x2 = bbox_int[2]
            y2 = bbox_int[3]
            embedding = face.normed_embedding  # The 512-dimensional vector
            list_emb = embedding.tolist()
            results = collection.search([list_emb], "embedding", search_params, limit=1)
            for result in results:
                print(f"Matched id: {result.ids[0]}, Distance: {result.distances[0]}")
                if result.distances[0] < 1.1:
                    name = result.ids[0]
                else:
                    name = "Unknow"
                cv2.putText(frame, str(result.distances[0]), (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (200, 255, 155), 2, cv2.LINE_AA)
                cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (200, 255, 155), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 255, 255), 1)
    else:
        cv2.putText(frame, 'No face', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (200, 255, 155), 2, cv2.LINE_AA)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Video', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




