import hashlib

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


def _chunk_id(chunk) -> str:
    source = chunk.metadata.get("source", "")
    page = chunk.metadata.get("page", "")
    raw = f"f{source}|{page}|{chunk.page_content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def split_and_import(loader, vector_db: Chroma) -> int:
    chunks = _text_splitter.split_documents(loader.load())
    if not chunks:
        return 0

    ids = [_chunk_id(chunk) for chunk in chunks]
    existing_ids = set(vector_db.get(ids=ids)["ids"])

    new_pairs = [(c, i) for c, i in zip(chunks, ids) if i not in existing_ids]
    if not new_pairs:
        print(f"No new chunks from {loader} (already ingested).")
        return 0

    new_chunks, new_ids = zip(*new_pairs)
    vector_db.add_documents(list(new_chunks), ids=list(new_ids))
    print(f"Ingested {len(new_chunks)} chunk(s) from {loader} "
          f"({len(chunks) - len(new_chunks)} already existed).")
    return len(new_chunks)


def ingest_wikipedia(query: str, vector_db: Chroma) -> int:
    return split_and_import(get_wikipedia_loader(query), vector_db)


def ingest_folder(folder_path: str, vector_db: Chroma) -> int:
    total = 0
    for filename, loader in iter_folder_loaders(folder_path):
        print(f"Loader for {filename}: {loader}")
        total += split_and_import(loader, vector_db)
    return total
