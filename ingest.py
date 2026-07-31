import os.path

from src.config import WIKIPEDIA_QUERY, PAESTUM_DIR, CILENTO_DIR
from src.vector_store import get_vector_db, ingest_wikipedia, ingest_folder


def main() -> None:
    vector_db = get_vector_db()
    total_chunks = 0

    print(f"Ingesting Wikipedia article (s) for '{WIKIPEDIA_QUERY}'...")
    try:
        total_chunks += ingest_wikipedia(WIKIPEDIA_QUERY, vector_db)
    except Exception as e:
        print(f"Skipping Wikipedia ingestion (error: {e})")

    for folder in (PAESTUM_DIR, CILENTO_DIR):
        if os.path.isdir(folder):
            print(f"Ingesting local folder: {folder}")
            total_chunks += ingest_folder(folder, vector_db)
        else:
            print(f"\nSkipping missing folder: {folder}")

    print(f"\nDone. {total_chunks} chunk(s) ingested in this run.")


if __name__ == "__main__":
    main()
