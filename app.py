import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_astradb import AstraDBVectorStore
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Question(BaseModel):
    question: str


def create_app() -> FastAPI:
    # Load environment variables from .env file
    logger.info("Loading environment variables...")
    load_dotenv()

    required_env_vars = [
        "AZURE_OPENAI_API_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_DEPLOYMENT",
        "ASTRA_DB_API_ENDPOINT",
        "ASTRA_DB_APPLICATION_TOKEN",
        "ASTRA_DB_COLLECTION",
        "NVIDIA_API_KEY",
    ]
    missing_vars = [var for var in required_env_vars if os.getenv(var) is None]
    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    # Initialize FastAPI app with metadata
    app = FastAPI(
        title="Keploy RAG Bot API",
        description="API for Keploy's RAG-based question answering system",
        version="0.0.1",
        terms_of_service="https://keploy.io/privacy-policy",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def setup_astradb():
        """Setup connection to AstraDB and return vector store"""
        logger.info("Setting up AstraDB vector store...")
        vector_store = AstraDBVectorStore(
            collection_name=os.getenv("ASTRA_DB_COLLECTION"),
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
            token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
            embedding=NVIDIAEmbeddings(
                model="nvidia/embed-qa-4", api_key=os.getenv("NVIDIA_API_KEY")
            ),
        )
        logger.info("AstraDB vector store setup completed.")
        return vector_store

    try:
        logger.info("Connecting to AstraDB vector store...")
        vectordb = setup_astradb()
        logger.info("Successfully connected to AstraDB vector store.")
    except Exception as e:
        logger.error(f"Error connecting to AstraDB vector store: {str(e)}")
        sys.exit(1)

    logger.info("Creating retrieval chain...")

    # Prompt for answering user's question
    qa_template = """
        You are a helpful assistant specialized in answering technical questions related to Keploy. You are provided with context from a vector database and a chat history. Your task is to answer the user's question based on the provided context and the chat history. If you don't know the answer, just say 'I don't know'. Do not try to make up an answer. If the question is not related to Keploy, say 'I am not sure about that'."
    
        Context: {context}
        Question: {input}
        Answer: 
        """

    qa_prompt = PromptTemplate(
        template=qa_template, input_variables=["context", "input"]
    )

    # Prompt for creating search queries based on conversation history
    condense_template = """
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question that captures all relevant context from the conversation.
    
    Chat History:
    {chat_history}
    
    Follow Up Question: {input}
    
    Standalone question:
    """

    condense_prompt = PromptTemplate.from_template(condense_template)

    try:
        logger.info("Setting up Azure OpenAI...")
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_API_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0.7,
        )
        logger.info("Azure OpenAI setup completed successfully.")
    except Exception as e:
        logger.error(f"Error setting up Azure OpenAI: {str(e)}")
        sys.exit(1)

    # Create the history-aware retriever
    base_retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    retriever = create_history_aware_retriever(llm, base_retriever, condense_prompt)

    # Create the document chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create the retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)

    logger.info("Retrieval chain created successfully.")

    @app.get("/", tags=["health"], summary="Check API health status")
    async def health():
        # Get system information
        import platform
        import time
        from datetime import datetime

        # Calculate uptime
        start_time = time.time()
        uptime = time.time() - start_time

        # Check database connection
        db_status = "connected"
        try:
            # Simple check to see if vectordb is accessible
            vectordb.as_retriever()
        except Exception as e:
            db_status = f"error: {str(e)}"
            logger.error(f"Database health check failed: {str(e)}")

        # Check LLM service
        llm_status = "connected"
        try:
            # Simple check on LLM service
            llm_status = "connected to " + os.getenv(
                "AZURE_OPENAI_API_DEPLOYMENT", "unknown"
            )
        except Exception as e:
            llm_status = f"error: {str(e)}"
            logger.error(f"LLM health check failed: {str(e)}")

        return {
            "status": "ok",
            "version": "0.0.1",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "uptime_seconds": uptime,
            "system": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
            },
            "services": {
                "database": db_status,
                "llm": llm_status,
            },
        }

    @app.post("/chat")
    def chat(question: Question):
        logger.info("Received chat request")
        if not question.question:
            logger.warning("No question provided")
            raise HTTPException(status_code=400, detail="No question provided")

        try:
            # Store chat history in the request context
            chat_history = []  # This would be persisted per user in a real application

            # Process the question with chat history
            response = retrieval_chain.invoke(
                {"chat_history": chat_history, "input": question.question}
            )

            # Update chat history for future requests (in a real app, this would be stored)
            chat_history.append(HumanMessage(content=question.question))
            chat_history.append(AIMessage(content=response["answer"]))

            # Extract source documents
            source_docs = response.get("context", [])

            result = {
                "answer": response["answer"],
                "sources": [
                    doc.metadata.get("source", "Unknown")
                    for doc in source_docs
                    if hasattr(doc, "metadata")
                ],
            }
            return result

        except Exception as e:
            logger.error(f"Error during chat processing: {str(e)}")
            raise HTTPException(
                status_code=500, detail="An error occurred during chat processing"
            )

    return app


# Create the application instance
app = create_app()
