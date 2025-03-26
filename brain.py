import os
import uuid
from io import BytesIO
from typing import Tuple, List, Any, Optional
import time
import re

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from astrapy import Database
from astrapy.constants import VectorMetric
from astrapy.info import CollectionVectorServiceOptions
from astrapy.exceptions import InsertManyException

from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "keploy_docs_test"
EMBEDDING_DIMENSION = 1024
MAX_VECTORIZE_LENGTH = 1000  # 1000 to stay under token limits
MAX_TOKENS = 475  # 512 token maximum


def parse_mdx(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    content = file.read().decode("utf-8")
    # You might want to add more sophisticated MDX parsing here
    return [content], filename


def parse_code_snippet(
    snippet: str, language: str, filename: str
) -> Tuple[List[str], str]:
    """Parse code snippets for embedding"""
    return [snippet], f"{filename}_{language}_snippet"


def text_to_docs(text: List[str], filename: str) -> List[Document]:
    if isinstance(text, str):
        text = [text]
    doc_chunks = []
    for i, page in enumerate(text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len,
        )
        chunks = text_splitter.split_text(page)
        for j, chunk in enumerate(chunks):
            # Calculate the start and end line numbers for this chunk
            start_line = page[: page.index(chunk)].count("\n") + 1
            end_line = start_line + chunk.count("\n")

            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk": j,
                    "source": f"{filename}:{start_line}-{end_line}",
                    "filename": filename,
                    "start_line": start_line,
                    "end_line": end_line,
                },
            )
            doc_chunks.append(doc)
    return doc_chunks


def connect_to_astra():
    """Connect to AstraDB using environment variables"""
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")

    if not token:
        raise ValueError("ASTRA_DB_APPLICATION_TOKEN environment variable is not set")

    if not api_endpoint:
        raise ValueError("ASTRA_DB_API_ENDPOINT environment variable is not set")

    astra_db = Database(token=token, api_endpoint=api_endpoint)

    return astra_db


def get_or_create_collection(astra_db):
    """Get existing collection or create a new one with NVIDIA embeddings"""
    try:
        collections = astra_db.list_collections()
        collection_exists = any(col.name == COLLECTION_NAME for col in collections)

        if collection_exists:
            print(f"* Using existing collection: {COLLECTION_NAME}")
            return astra_db.get_collection(COLLECTION_NAME)

        # Collection doesn't exist, create it
        print(f"* Creating new collection: {COLLECTION_NAME}")
        collection = astra_db.create_collection(
            COLLECTION_NAME,
            metric=VectorMetric.COSINE,
            service=CollectionVectorServiceOptions(
                provider="nvidia",
                model_name="NV-Embed-QA",
            ),
        )
        print(f"* Successfully created collection: {COLLECTION_NAME}")
        return collection
    except Exception as e:
        print(f"* Error during collection creation/retrieval: {e}")

        try:
            print(f"* Attempting to use get_collection method for: {COLLECTION_NAME}")
            return astra_db.get_collection(COLLECTION_NAME)
        except AttributeError:
            print(
                "* get_collection method not found, checking for alternative methods..."
            )
            try:
                print(f"* Trying find_collection method for: {COLLECTION_NAME}")
                return astra_db.find_collection(COLLECTION_NAME)
            except AttributeError:
                pass

        raise ValueError(f"Failed to access collection: {e}")


def truncate_to_token_limit(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """Truncate text to approximately stay under token limit.

    This is a simple estimation since we don't have access to the actual tokenizer.
    On average, 1 token is roughly 4 characters in English text.
    """
    # Simple regex to split on word boundaries
    words = re.findall(r"\b\w+\b|[^\w\s]|\s+", text)

    result = ""
    estimated_token_count = 0

    for word in words:
        word_token_estimate = max(1, round(len(word) / 4))

        if estimated_token_count + word_token_estimate > max_tokens:
            break

        result += word
        estimated_token_count += word_token_estimate

    return result


def store_documents(collection, documents: List[Document]) -> None:
    """Store documents in AstraDB collection with truncated vectorization text"""
    vector_docs = []
    success_count = 0
    error_count = 0
    total_docs = len(documents)

    print(f"Preparing to store {total_docs} documents in AstraDB...")

    for doc in documents:
        # unique ID for the document
        doc_id = str(uuid.uuid4())

        # first part of content after applying token-aware truncation
        vectorize_text = truncate_to_token_limit(doc.page_content)

        # summary from metadata if available, but be cautious of token limits
        if doc.metadata and "filename" in doc.metadata:
            source_info = f"Source: {doc.metadata.get('source', 'unknown')}"
            # Only add source if we have room for it
            combined_text = f"{source_info}\n{vectorize_text}"
            if len(combined_text) <= MAX_VECTORIZE_LENGTH:
                vectorize_text = combined_text
            else:
                shorter_text = truncate_to_token_limit(
                    doc.page_content, MAX_TOKENS - 20
                )  # Reserve ~20 tokens for source
                vectorize_text = f"{source_info}\n{shorter_text}"

        vector_doc = {
            "_id": doc_id,
            "content": doc.page_content,
            "metadata": doc.metadata,
            "$vectorize": vectorize_text,  # Use token-aware truncated content for vectorization
        }

        vector_docs.append(vector_doc)

    # Batch insert documents in smaller batches to handle potential errors
    batch_size = 20
    batches = (
        total_docs + batch_size - 1
    ) // batch_size  # Calculate total number of batches

    for i in range(0, len(vector_docs), batch_size):
        batch = vector_docs[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"Processing batch {batch_num}/{batches} ({len(batch)} documents)...")

        try:
            inserted = collection.insert_many(batch)
            success_count += len(inserted.inserted_ids)
            print(f"  ✓ Successfully inserted {len(inserted.inserted_ids)} documents")
        except InsertManyException as e:
            print(f"  ⚠ Error inserting batch {batch_num}: {e}")

            # Check if it's a token size error
            error_str = str(e)
            if "exceeds maximum allowed token size" in error_str:
                print("  ⚠ Token size limit exceeded. Retrying with smaller content...")
                for doc in batch:
                    if "$vectorize" in doc:
                        doc["$vectorize"] = truncate_to_token_limit(
                            doc["$vectorize"], MAX_TOKENS - 50
                        )

            for j, doc in enumerate(batch):
                try:
                    collection.insert_one(doc)
                    success_count += 1
                    if j % 5 == 0:
                        print(
                            f"  - Inserted {j + 1}/{len(batch)} individual documents..."
                        )
                except Exception as single_error:
                    error_count += 1
                    source = doc.get("metadata", {}).get("source", "unknown")
                    print(
                        f"  ✗ Failed to insert document from {source}: {single_error}"
                    )

                    if (
                        "exceeds maximum allowed token size" in str(single_error)
                        and "$vectorize" in doc
                    ):
                        vec_text = doc["$vectorize"]
                        print(
                            f"    - Vectorize text length: {len(vec_text)} chars, est. {len(vec_text) / 4:.0f} tokens"
                        )

    print(
        f"Indexing complete: {success_count} succeeded, {error_count} failed out of {total_docs} documents"
    )

def get_index_for_mdx(mdx_files, mdx_names):
    """Create or update index for MDX files using AstraDB"""
    print(f"Creating index for {len(mdx_files)} document files...")

    astra_db = connect_to_astra()
    collection = get_or_create_collection(astra_db)

    documents = []
    for i, (mdx_file, mdx_name) in enumerate(zip(mdx_files, mdx_names)):
        print(f"Processing document {i + 1}/{len(mdx_files)}: {mdx_name}")
        text, filename = parse_mdx(BytesIO(mdx_file), mdx_name)
        doc_chunks = text_to_docs(text, filename)
        documents.extend(doc_chunks)
        print(f"  - Generated {len(doc_chunks)} chunks from {filename}")

    store_documents(collection, documents)

    return collection


def update_index_with_code(
    code_snippet: str, language: str, filename: str
) -> Optional[Any]:
    """Add a code snippet to the index"""
    if not code_snippet:
        return None

    astra_db = connect_to_astra()
    collection = get_or_create_collection(astra_db)

    text, snippet_name = parse_code_snippet(code_snippet, language, filename)
    documents = text_to_docs(text, snippet_name)

    store_documents(collection, documents)

    return collection

if __name__ == "__main__":
    """Main function to test the functionality"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python brain.py [index|search|add-code]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "index":
        if len(sys.argv) < 3:
            print("Usage: python brain.py index <directory> [--dry-run]")
            sys.exit(1)

        import glob

        directory = sys.argv[2]
        dry_run = "--dry-run" in sys.argv

        start_time = time.time()
        mdx_files = []
        mdx_names = []
        skipped_files = []

        file_count = 0
        for file_pattern in ["**/*.mdx", "**/*.md"]:
            pattern = f"{directory}/{file_pattern}"
            print(f"Searching for files matching: {pattern}")

            for file_path in glob.glob(pattern, recursive=True):
                file_count += 1
                try:
                    with open(file_path, "rb") as f:
                        mdx_files.append(f.read())
                        mdx_names.append(
                            file_path
                        )  # Store full path for better identification
                except Exception as e:
                    skipped_files.append((file_path, str(e)))

        print(f"Found {file_count} files, loaded {len(mdx_files)} successfully")

        if skipped_files:
            print(f"Warning: Skipped {len(skipped_files)} files due to errors:")
            for file_path, error in skipped_files[:5]:  # Show first 5 errors
                print(f"  - {file_path}: {error}")
            if len(skipped_files) > 5:
                print(f"  ... and {len(skipped_files) - 5} more")

        if dry_run:
            print("Dry run completed. No documents were indexed.")
        else:
            get_index_for_mdx(mdx_files, mdx_names)

        elapsed_time = time.time() - start_time
        print(f"Total processing time: {elapsed_time:.2f} seconds")

    else:
        print(f"Unknown command: {command}")
        print("Available commands: index")
        sys.exit(1)
