import cv2
import numpy as np
import insightface
import faiss
import pandas as pd
import cv2
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

################### INITIAL FAISS #####################
list_name = []
face_index = []
df = pd.read_csv("faces.csv")
face_index = faiss.IndexFlatL2(512)
for _, row in df.iterrows():
    embedding = row.iloc[1:513].to_numpy().astype('float32')
    embedding = np.ascontiguousarray(embedding.reshape(1, 512))
    one_name = row.iloc[0]
    list_name.append(one_name)
    face_index.add(embedding)
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
            embedding = np.array(embedding, dtype=np.float32).reshape(-1, 512)
            dis, result = face_index.search(embedding, k=1)
            if dis and result is not None:
                dist = float(dis[0][0:])
                index = int(result[0][0:])
                if dist < 1.1:
                    name = list_name[index]
                    name = name.rstrip(".jpg")
                else:
                    name = "Unknow"
                # cv2.putText(frame, str(dist), (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (200, 255, 155), 2, cv2.LINE_AA)
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




