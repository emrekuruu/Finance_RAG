import pinecone

class VectorDBHandler:
    def __init__(self, index_name: str):
        """
        Initializes the handler to connect with the Pinecone index.
        
        Args:
        - index_name (str): The name of the Pinecone index to interact with.
        """
        self.index = pinecone.Index(index_name)
    
    def upsert_vector(self, vector_id: str, embedding: list):
        """
        Upserts (inserts or updates) a vector with a given ID and embedding.
        
        Args:
        - vector_id (str): Unique ID for the vector.
        - embedding (list): The embedding vector to store.
        """
        self.index.upsert(vectors=[(vector_id, embedding)])
    
    def query_vector(self, query_embedding: list, top_k: int = 5):
        """
        Queries the Pinecone index for the most similar vectors to the query embedding.
        
        Args:
        - query_embedding (list): The embedding to query for.
        - top_k (int): Number of top results to retrieve.
        
        Returns:
        - List of dictionaries containing the IDs and scores of the top results.
        """
        return self.index.query(queries=[query_embedding], top_k=top_k)['matches']
    
    def delete_vector(self, vector_id: str):
        """
        Deletes a vector from the Pinecone index.
        
        Args:
        - vector_id (str): The unique ID of the vector to delete.
        """
        self.index.delete(ids=[vector_id])
    
    def fetch_vector(self, vector_id: str):
        """
        Fetches a vector from the index by its ID.
        
        Args:
        - vector_id (str): The unique ID of the vector to fetch.
        
        Returns:
        - A dictionary containing the vector's data.
        """
        return self.index.fetch(ids=[vector_id])


class VectorDatabase:
    def __init__(self, api_key: str, environment: str):
        """
        Initializes Pinecone client.
        
        Args:
        - api_key (str): Pinecone API key.
        - environment (str): Pinecone environment (e.g., 'us-west1-gcp').
        """
        pinecone.init(api_key=api_key, environment=environment)
    
    def start_db(self, index_name: str, dimension: int):
        """
        Starts a Pinecone index (vector database) if it doesn't exist and returns a handler object.
        
        Args:
        - index_name (str): Name of the index to create or connect to.
        - dimension (int): Dimensionality of the vectors to store.
        
        Returns:
        - VectorDBHandler: An object that handles read/write operations for the index.
        """
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=dimension)
        return VectorDBHandler(index_name)
