- Project face recognition using: 

RetinaFace to make face with mask.

InsightFace to convert face to vector 512 dims.

faiss-cpu to search euclid distance between vector face detect with many vectors in data csv.

milvus vectordatabase to search in database

- To use project with faiss similarity search:

pip install -r requirements.txt

put images .jpg with full face in ImageFolder

run convert_images_folder_to_csv.py to wear face mask to face and covert face with no mask and face with mask to vector 512 dims.

run inference.py to detect face with webcam, change video_path to custom your input (rtsp link or link video)

- Use milvus vectordatabase