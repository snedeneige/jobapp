import os
import numpy as np
import pandas as pd
import logging
from openai import OpenAI


class Chat:
    """
    A class to manage chat interactions with the OpenAI API.
    
    Attributes:
        key (str): The API key from the environment.
        client (OpenAI): The OpenAI client instance.
        messages (list[dict]): A list of messages in the conversation.
    """

    def __init__(self, instructions: str = "You're a helpful AI assistant.") -> None:
        """
        Initialize the Chat object with an initial developer instruction.
        
        Parameters:
            instructions (str): The starting instruction for the chat conversation.
        """
        self.key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI()
        self.messages = [{"role": "developer", "content": instructions}]
        
    def add_assistant_message(self, message: str) -> None:
        """
        Add a message from the assistant to the conversation history.
        
        Parameters:
            message (str): The message to add to the conversation.
        """
        self.messages.append({"role": "assistant", "content": message})

    def add_user_message(self, message: str) -> None:
        """
        Add a message from the user to the conversation history.
        
        Parameters:
            message (str): The message to add to the conversation.
        """
        self.messages.append({"role": "user", "content": message})

    def query_response(self, add_to_history : bool = True) -> str:
        chats = self.messages.copy()
        chat_completion = self.client.chat.completions.create(
            messages=chats,
            model="gpt-4o-mini",
        )
        
        response = chat_completion.choices[0].message.content

        if add_to_history:
            self.add_assistant_message(response)
        
        return response


class Vectorizer:
    """
    A class for converting text into numerical vectors using the OpenAI embeddings API.
    
    Attributes:
        vector_length (int): The desired length of the output vectors.
        key (str): The API key from the environment.
        client (OpenAI): The OpenAI client instance.
    """

    def __init__(self, vector_length: int) -> None:
        """
        Initialize the Vectorizer with a specific vector length.
        
        Parameters:
            vector_length (int): The length of the vectors to be produced.
        """
        self.vector_length = vector_length
        self.key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI()

    def vectorize(self, text: str | list[str]) -> np.ndarray | pd.DataFrame | None:
        """
        Convert a single text string or a list of texts into their vector representations.
        
        Parameters:
            text (str | list[str]): A single text string or a list of text strings.
            
        Returns:
            np.ndarray: If a single text is provided.
            pd.DataFrame: If a list of texts is provided.
            None: If an error occurs or the input type is invalid.
        """
        try:
            if isinstance(text, str):
                return self._vectorize_single(text)
            elif isinstance(text, list):
                return self.vectorize_list(text)
            else:
                logging.error("Invalid input type for vectorize: expected str or list[str].")
                return None
        except Exception as e:
            logging.exception("An error occurred during vectorization.")
            return None

    def _vectorize_single(self, text: str) -> np.ndarray:
        """
        Convert a single text string into its vector representation.
        
        Parameters:
            text (str): The text to vectorize.
            
        Returns:
            np.ndarray: The numerical vector corresponding to the input text.
        """
        response = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            dimensions=self.vector_length
        )
        embedding_list = response.data[0].embedding
        return np.array(embedding_list)

    def vectorize_list(self, text_list: list[str]) -> pd.DataFrame:
        """
        Convert a list of text strings into a DataFrame of vector representations.
        
        Parameters:
            text_list (list[str]): The texts to vectorize.
            
        Returns:
            pd.DataFrame: A DataFrame where each row corresponds to the vector of a text.
        """
        response = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=text_list,
            dimensions=self.vector_length
        )
        embeddings = response.data
        vectors = {text: np.array(embedding.embedding) for text, embedding in zip(text_list, embeddings)}
        return pd.DataFrame(vectors).T