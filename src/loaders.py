import os.path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader, WikipediaLoader, \
    DirectoryLoader

LOADER_CLASSES = {
    "docx": Docx2txtLoader,
    "pdf": PyPDFLoader,
    "txt": TextLoader,
}


def get_loader(filename: str):
    _, file_extension = os.path.splitext(filename)
    file_extension = file_extension.lstrip(".")

    loader_class = LOADER_CLASSES.get(file_extension)
    if loader_class is None:
        raise ValueError(f"No loader available for file extension '{file_extension}'")
    return loader_class(filename)


def iter_folder_loaders(folder_path: str):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue
        try:
            yield filename, get_loader(file_path)
        except ValueError as e:
            print(f"Skipping {filename}: {e}")


def get_wikipedia_loader(query: str) -> WikipediaLoader:
    return WikipediaLoader(query=query)


def get_directory_loader(folder_path: str, pattern: str = "**/*.{docx,pdf,txt}"):
    return DirectoryLoader(folder_path, glob=pattern)
