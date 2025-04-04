import logging
from rag import Rag
from sopenai import Chat

logging.basicConfig(level=logging.INFO)

class JobAgent:
    """
    An Agent that integrates Retrieval-Augmented Generation (RAG) with chat capabilities.
    It processes a user question by retrieving relevant texts from documents and incorporating
    these texts into the conversation before querying the chat system.
    """

    def __init__(self, job_id : str, job_description : str = None, agent_description : str = None) -> None:
        """
        Initialize the Agent with instances of Rag and Chat, and a set to track already retrieved documents.
        """
        self.job_id = job_id
        self.rag = Rag()
        if agent_description:
            self.chat = Chat(agent_description)
        else:
            self.chat = Chat()
        self.retrieved_docs: set[str] = set()

        if job_description:
            self.chat.add_assistant_message(f"This is the job description: {job_description}")
            logging.info("Added job description to chat for %s-agent", job_id)

    def process_question(self, question: str) -> str:
        """
        Process the user question by:
        
          1. Retrieving relevant texts from documents using RAG.
          2. Filtering out documents that have already been provided.
          3. Adding context messages (assistant messages) with new retrieved texts.
          4. Appending the user question and obtaining a response from the chat system.
        
        Parameters:
            question (str): The user's question.
        
        Returns:
            str: The response generated by the chat system.
        """
        try:
            # Retrieve relevant texts based on the user's question.
            relevant_texts = self.rag.retrieve_relevant_texts(question)
            if not isinstance(relevant_texts, dict):
                logging.error("Expected to retrieve a dictionary of relevant texts for %s-agent, but got: '%s'", self.job_id, type(relevant_texts))
                return "Error: Unable to process your request at this time."
            
            # Filter out texts that have already been provided.
            new_relevant_texts = {doc: text for doc, text in relevant_texts.items() if doc not in self.retrieved_docs}
            
            if new_relevant_texts:
                self.chat.add_assistant_message(
                    "For the next user question, you may want to consider the following information:"
                )
            
            for doc, text in new_relevant_texts.items():
                message = f'>>>>>>>> Retrieved text from {doc}: {text} <<<<<<<<'
                self.chat.add_assistant_message(message)
                logging.info("%s-agent retrieved text for document: '%s'", self.job_id, doc)
                self.retrieved_docs.add(doc)
            
            self.chat.add_user_message(question)
            
            response = self.chat.query_response()
            logging.info("Chat response received from OpenAI for %s-agent: '%s'", self.job_id, response)
            return response
        
        except Exception as e:
            logging.exception("An error occurred for %s-agent while processing the question.", self.job_id)
            return "An error occurred while processing your question."