from abc import ABC, abstractmethod
import torch
import torch.nn as nn

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

class CrossEncoder(Encoder):

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.eval_mode = False

    def encode(self, query: str, document: str):
        # Concatenate the query and document
        combined_text = query + " [SEP] " + document
        # Tokenize the combined input
        inputs = self.tokenizer(combined_text, return_tensors='pt', truncation=True, padding=True)
        
        if self.eval_mode:
            with torch.no_grad():
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)
            
        # Extract embeddings from the [CLS] token
        return outputs.last_hidden_state[:, 0, :].squeeze()

    def encode_batch(self, queries: list, documents: list):
        # Ensure queries and documents are paired
        assert len(queries) == len(documents), "Queries and documents must have the same length."
        combined_texts = [q + " [SEP] " + d for q, d in zip(queries, documents)]
        # Tokenize the combined inputs
        inputs = self.tokenizer(combined_texts, return_tensors='pt', truncation=True, padding=True)
        
        if self.eval_mode:
            with torch.no_grad():
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)
        
        # Extract embeddings from the [CLS] token for each input
        return outputs.last_hidden_state[:, 0, :]
    
    def parameters(self):
        return list(self.model.parameters()) 
    
    def train(self):
        self.train_mode = False
        self.model.train()
    
    def eval(self):
        self.train_mode = True
        self.model.eval()

    