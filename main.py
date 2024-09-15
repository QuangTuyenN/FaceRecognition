import insightface
import cv2
import numpy as np

# Load the InsightFace model
model = insightface.app.FaceAnalysis(allowed_modules=['detection', 'recognition'])
model.prepare(ctx_id=0)  # -1 for CPU, use GPU ID if available (e.g., 0 for GPU)

# Load the image and convert it to RGB
img_path = 'anh3.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect faces and extract features (embeddings)
faces = model.get(img)

# If a face is detected, extract the 512-dimensional vector
if len(faces) > 0:
    face = faces[0]  # Get the first detected face
    embedding = face.normed_embedding  # The 512-dimensional vector
    print("512-dimensional embedding vector:")
    print(embedding)
else:
    print("No face detected")
