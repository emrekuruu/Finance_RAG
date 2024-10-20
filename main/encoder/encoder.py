from abc import ABC, abstractmethod
import torch

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


class BiEncoder(Encoder):

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def encode(self, text: str):
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Extract the embeddings from the last hidden state of the [CLS] token
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    def encode_batch(self, texts: list):
        # Tokenize a batch of inputs
        inputs = self.tokenizer(texts, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Extract embeddings from the last hidden state of the [CLS] token for each input
        return outputs.last_hidden_state[:, 0, :].numpy()
