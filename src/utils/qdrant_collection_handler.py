from qdrant_client import QdrantClient

# Connect to Qdrant
client = QdrantClient("localhost", port=6333)

# Get all collections
collections = client.get_collections().collections

# Filter out the "document" collection
collections_to_delete = [col.name for col in collections if col.name != "documents"]

# Delete each collection
for collection in collections_to_delete:
    client.delete_collection(collection)
    print(f"Deleted collection: {collection}")

print("All collections except 'document' have been deleted.")
