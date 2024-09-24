import pandas as pd
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

df = pd.read_csv('faces.csv')

ids = df['id'].tolist()
vectors = df.iloc[:, 1:].values

# connect to milvus server run on port 19530
connections.connect("default", host="localhost", port="19530")

# set schema for collection
id_field = FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True)
embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
schema = CollectionSchema(fields=[id_field, embedding_field], description="Face embeddings collection")

# make collection
collection = Collection(name="face_embeddings", schema=schema)

# prepare data to insert
entities = [ids, vectors]

# Insert data to collection
collection.insert(entities)

# call flush to ensure data is written to Milvus
collection.flush()

print(f"Inserted {len(ids)} records into Milvus")

# # prepare new vector to search
# query_vector = [0.1, 0.2, ..., 0.5]
#
# # find 5 vector
# # In Milvus, the nprobe parameter is an optimization parameter
# # for ANN (Approximate Nearest Neighbor) search,
# # which involves a trade-off between search speed and accuracy.
# # nprobe determines the number of clusters that will be traversed
# # during the search process. Milvus uses the IVF (Inverted File Index)
# # technique to divide the vector space into many different regions.
# # When searching for a vector, instead of going through all the regions
# # (which is very slow), Milvus only goes through the few regions that have
# # the best potential to contain matches.
# # The higher the nprobe value, the more accurate the search, but also slower.
# # The lower the nprobe value, the faster the search, but the accuracy may decrease.
# search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
# results = collection.search([query_vector], "embedding", search_params, limit=5)
#
# for result in results:
#     print(f"Matched id: {result.ids[0]}, Distance: {result.distances[0]}")




