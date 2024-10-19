from abc import ABC, abstractmethod

class Encoder(ABC):
    """
    Abstract base class for Encoders.
    This class defines the interface for encoding queries and document chunks.
    """
    
    @abstractmethod
    def encode(self, text: str):
        """
        Abstract method to encode a single text input into a vector representation.
        
        Args:
        - text (str): The input text to encode.
        
        Returns:
        - List of floats representing the embedding.
        """
        pass

    @abstractmethod
    def encode_batch(self, texts: list):
        """
        Abstract method to encode a batch of text inputs into vector representations.
        
        Args:
        - texts (list of str): A list of input texts to encode.
        
        Returns:
        - List of embeddings where each embedding corresponds to a text input.
        """
        pass
