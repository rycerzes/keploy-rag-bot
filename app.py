import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_astradb import AstraDBVectorStore
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
logger.info("Loading environment variables...")
load_dotenv()

required_env_vars = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "ASTRA_DB_API_ENDPOINT",
    "ASTRA_DB_APPLICATION_TOKEN",
    "ASTRA_DB_COLLECTION",
    "NVIDIA_API_KEY",
]
missing_vars = [var for var in required_env_vars if os.getenv(var) is None]
if missing_vars:
    logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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

logger.info("Creating conversational chain...")
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
    max_messages=10,
)

template = """
    You are a helpful assistant specialized in answering technical questions related to Keploy. You are provided with context from a vector database and a chat history. Your task is to answer the user's question based on the provided context and the chat history. If you don't know the answer, just say 'I don't know'. Do not try to make up an answer. If the question is not related to Keploy, say 'I am not sure about that'."

    Context: {context}
    Question: {question}
    Answer: 
    """

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

try:
    logger.info("Setting up Azure OpenAI...")
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_API_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.7,
    )
    logger.info("Azure OpenAI setup completed successfully.")
except Exception as e:
    logger.error(f"Error setting up Azure OpenAI: {str(e)}")
    sys.exit(1)

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    return_source_documents=True,
    verbose=False,
    combine_docs_chain_kwargs={"prompt": prompt},
)
logger.info("Conversational chain created successfully.")


class Question(BaseModel):
    question: str


@app.post("/chat")
def chat(question: Question):
    logger.info("Received chat request")
    if not question.question:
        logger.warning("No question provided")
        raise HTTPException(status_code=400, detail="No question provided")

    try:
        search_results = vectordb.similarity_search(question.question, k=3)
        context = "\n".join([doc.page_content for doc in search_results])

        response = conversation_chain({"question": question.question})

        # logger.info(f"Response from conversation chain: {response}")

        result = {
            "answer": response["answer"],
            "sources": [
                doc.metadata.get("source", "Unknown")
                for doc in response.get("source_documents", [])
            ],
        }
        return result

    except Exception as e:
        logger.error(f"Error during chat processing: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An error occurred during chat processing"
        )


if __name__ == "__main__":
    logger.info("Starting FastAPI app...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.info("FastAPI app started.")
