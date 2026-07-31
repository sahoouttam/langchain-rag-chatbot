from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from src.loaders import get_wikipedia_loader, iter_folder_loaders

_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)

_embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vector_db() -> Chroma:
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=_embeddings_model,
        persist_directory=CHROMA_PERSIST_DIR
    )


def split_and_import(loader, vector_db: Chroma) -> int:
    chunks = _text_splitter.split_documents(loader.load())
    if not chunks:
        return 0
    vector_db.add_documents(chunks)
    print(f"Ingested {len(chunks)} chunk(s) from {loader}")
    return len(chunks)


def ingest_wikipedia(query: str, vector_db: Chroma) -> int:
    return split_and_import(get_wikipedia_loader(query), vector_db)


def ingest_folder(folder_path: str, vector_db: Chroma) -> int:
    total = 0
    for filename, loader in iter_folder_loaders(folder_path):
        print(f"Loader for {filename}: {loader}")
        total += split_and_import(loader, vector_db)
    return total
