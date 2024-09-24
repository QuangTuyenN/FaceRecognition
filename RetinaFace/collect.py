import numpy as np
from pymilvus import Collection, connections

# Kết nối đến Milvus
connections.connect("default", host="localhost", port="19530")

# Lấy collection
collection = Collection("face_embeddings")

# Truy vấn một số vector từ collection
results = collection.query(name="face_embeddings", limit=100)  # Giới hạn 100 vector