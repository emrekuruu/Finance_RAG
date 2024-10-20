import pinecone
from pinecone import Pinecone, ServerlessSpec

class VectorDBHandler:
    def __init__(self, index_name: str, pinecone_instance: Pinecone):
        """
        Initializes the handler to connect with the Pinecone index.
        
        Args:
        - index_name (str): The name of the Pinecone index to interact with.
        - pinecone_instance (Pinecone): The initialized Pinecone instance.
        """
        self.index = pinecone_instance.Index(index_name)
    
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
        - query_embedding (list): The embedding to query for (should be a 1D list of floats).
        - top_k (int): Number of top results to retrieve.
        
        Returns:
        - List of dictionaries containing the IDs and scores of the top results.
        """
        # Query the Pinecone index with a single query embedding
        return self.index.query(vector=query_embedding, top_k=top_k)['matches']
    
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

    def delete_all(self):
        """
        Deletes all vectors from the Pinecone index.
        """
        self.index.delete(delete_all=True)

class VectorDatabase:
    def __init__(self, api_key: str):
        """
        Initializes Pinecone client.
        
        Args:
        - api_key (str): Pinecone API key.
        """
        self.pc = Pinecone(api_key=api_key)
    
    def start_db(self, index_name: str, dimension: int, cloud: str, region: str):
        """
        Starts a Pinecone index (vector database) if it doesn't exist and returns a handler object.
        
        Args:
        - index_name (str): Name of the index to create or connect to.
        - dimension (int): Dimensionality of the vectors to store.
        - cloud (str): Cloud provider (e.g., 'aws').
        - region (str): Cloud region (e.g., 'us-west-1').
        
        Returns:
        - VectorDBHandler: An object that handles read/write operations for the index.
        """
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='dotproduct',
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )
        return VectorDBHandler(index_name, self.pc)