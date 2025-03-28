import os
from dotenv import load_dotenv
import sys

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_astradb import AstraDBVectorStore, utils
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# Load environment variables
load_dotenv()

# Check for required environment variables
required_env_vars = [
    "GROQ_API_KEY",
    "ASTRA_DB_API_ENDPOINT",
    "ASTRA_DB_APPLICATION_TOKEN",
    "ASTRA_DB_COLLECTION",
    "NVIDIA_API_KEY",
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    print(f"Missing required environment variables: {', '.join(missing_vars)}")
    print("Please set them in a .env file or in your environment.")
    sys.exit(1)


# Configure AstraDB
def setup_astradb():
    """Setup connection to AstraDB and return vector store"""
    vector_store = AstraDBVectorStore(
        collection_name=os.getenv("ASTRA_DB_COLLECTION"),
        api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
        embedding=NVIDIAEmbeddings(
            model="nvidia/embed-qa-4", api_key=os.getenv("NVIDIA_API_KEY")
        ),
        setup_mode=utils.astradb.SetupMode.OFF,  # the collection will be used as-is
    )
    return vector_store


# Configure Groq LLM
def setup_llm():
    """Setup Groq LLM"""
    model_name = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=model_name,
        temperature=0.5,
        max_tokens=1024,
    )
    return llm


# Setup RAG pipeline
def setup_rag_chain(vector_store, llm):
    """Configure the RAG chain with retriever and LLM"""
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    template = """
    You are a helpful assistant that provides accurate information based on the given context.
    
    Context:
    {context}
    
    Chat History:
    {chat_history}
    
    User Question: {question}
    
    Provide a detailed and helpful answer based on the context provided. If the context doesn't contain 
    relevant information to answer the question, just say that you don't have enough information.
    
    At the end of your answer, list the sources used to generate the response in a section titled "Sources:".
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Remove the deprecated ConversationBufferMemory and use a simple function

    def format_docs(docs):
        formatted_content = "\n\n".join(doc.page_content for doc in docs)
        # Extract source information from documents
        sources = []
        for doc in docs:
            if hasattr(doc, "metadata") and "source" in doc.metadata:
                source = doc.metadata["source"]
                if source not in sources:
                    sources.append(source)

        # Add sources to the formatted content
        if sources:
            source_info = "\n\nDocument sources: " + ", ".join(sources)
            return formatted_content + source_info
        return formatted_content

    def get_formatted_chat_history(chat_history):
        return "\n".join([f"{m.type}: {m.content}" for m in chat_history])

    def rag_chain(question, chat_history):
        # Format the chat history
        formatted_chat_history = get_formatted_chat_history(chat_history)
        
        # Get documents from the retriever
        docs = retriever.invoke(question)
        formatted_docs = format_docs(docs)
        
        # Run through the prompt, LLM, and parser
        chain_input = {
            "context": formatted_docs,
            "chat_history": formatted_chat_history,
            "question": question
        }
        response = prompt.invoke(chain_input)
        result = llm.invoke(response)
        return StrOutputParser().invoke(result)

    return rag_chain


def main():
    print("Setting up RAG Chatbot with Groq and AstraDB...")

    # Setup components
    vector_store = setup_astradb()
    llm = setup_llm()
    rag_chain = setup_rag_chain(vector_store, llm)
    
    # Initialize chat history
    chat_history = []

    print("RAG Chatbot is ready! Type 'exit' to quit.")

    # Simple chat loop
    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break

        try:
            # Get response from RAG chain
            response = rag_chain(user_input, chat_history)

            # Update chat history directly with message objects
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response))

            # Print response
            print(f"\nBot: {response}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
