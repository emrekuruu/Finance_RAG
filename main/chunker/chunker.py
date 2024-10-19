from abc import ABC, abstractmethod

class Chunker(ABC):
    """
    Abstract base class for document chunking.
    Each chunker must implement the chunk_document method.
    """
    
    @abstractmethod
    def chunk_document(self, document: str):
        """
        Abstract method to chunk a document.
        
        Args:
        - document (str): The document text to be chunked.
        
        Returns:
        - List of document chunks.
        """
        pass
