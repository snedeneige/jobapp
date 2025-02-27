import pandas as pd
from sopenai import Vectorizer
from pathlib import Path
import logging

class Rag:
    """
    A Retrieval-Augmented Generation (RAG) class that loads and vectorizes documents,
    then computes similarity scores between input text and these documents.
    """

    def __init__(self, documents_dir: str = "documents"):
        """
        Initialize the RAG system by loading document texts, vectorizing them, and preparing
        the necessary data structures.

        Parameters:
            documents_dir (str): The directory containing document text files.
        """
        self.vectorizer = Vectorizer(256)
        
        # load all documents from the specified directory:
        
        files = Path(documents_dir).glob("*.txt")

        self.document_names = [file.stem for file in files]

        self.documents: dict[str, str] = {}
        documents_text = []
        
        # Use pathlib for robust file handling.
        base_path = Path(documents_dir)
        for document in self.document_names:
            file_path = base_path / f"{document}.txt"
            try:
                with file_path.open("r", encoding="utf-8") as file:
                    text = file.read()
                    documents_text.append(text)
                    self.documents[document] = text
            except FileNotFoundError:
                logging.warning("File not found: '%s'", file_path)
            except Exception as e:
                logging.warning("Error reading %s: '%s'", file_path, e)

        if not documents_text:
            raise ValueError("No documents were loaded. Please check the document files.")

        # Vectorize the loaded documents.
        vocab_vectors_df: pd.DataFrame = self.vectorizer.vectorize(documents_text)

        # Rename DataFrame indices to match the document names.
        vocab_vectors_df.index = list(self.documents.keys())
        self.vocab_vectors_df: pd.DataFrame = vocab_vectors_df

    def retrieve_text(self, document_name: str) -> str | None:
        """
        Retrieve the text content of a specified document.

        Parameters:
            document_name (str): The name of the document to retrieve.

        Returns:
            str: The content of the document, or None if not found.
        """
        return self.documents.get(document_name)

    def compute_doc_similarities(self, text: str) -> dict[str, float]:
        """
        Compute similarity scores between the input text and each loaded document.

        Parameters:
            text (str): The input text for which to compute similarity scores.

        Returns:
            dict[str, float]: A mapping from document names to their similarity scores.
        """
        text_vector = self.vectorizer.vectorize(text)
        similarities = self.vocab_vectors_df.dot(text_vector.T).to_dict()
        return similarities

    def get_most_relevant_doc(self, text: str) -> str | None:
        """
        Identify the document that is most similar to the input text.

        Parameters:
            text (str): The input text to compare.

        Returns:
            str: The name of the most similar document found, or None if no similarities.
        """
        similarities = self.compute_doc_similarities(text)
        if not similarities:
            return None
        most_similar = max(similarities, key=similarities.get)
        return most_similar
    
    def retrieve_most_relevant_text(self, text: str) -> str | None:
        """
        Retrieve the text content of the document most similar to the input text.

        Parameters:
            text (str): The input text to compare.

        Returns:
            str: The content of the most similar document, or None if no similarities.
        """
        most_similar_doc = self.get_most_relevant_doc(text)
        if most_similar_doc:
            return self.retrieve_text(most_similar_doc)
        return None
    
    def get_relevant_docs(self, text: str) -> dict[str, float]:
        """
        Retrieve the most relevant documents based on the input text.

        Parameters:
            text (str): The input text to compare.

        Returns:
            dict[str, float]: A mapping from document names to their relevance scores.
        """
        similarities = self.compute_doc_similarities(text)
        return {doc: score for doc, score in similarities.items() if score > 0.2}
    
    def retrieve_relevant_texts(self, text: str) -> dict[str, str]:
        """
        Retrieve the text content of the most relevant documents based on the input text.

        Parameters:
            text (str): The input text to compare.

        Returns:
            dict[str, str]: A mapping from document names to their content.
        """
        relevant_docs = self.get_relevant_docs(text)
        return {doc: self.retrieve_text(doc) for doc in relevant_docs.keys()}