from pathlib import Path

def load_document(pdf_path: Path) -> str:
    """
    This function loads the documents from given path.

    Args:
        pdf_path: Relative path to the pdf file.
    """
    pass

def clean_text(text: str) -> str:
    """
    This function removes the noise like page numbers, extra whitespaces, header, artifacts from PDF extracted corpus.

    Args:
        text: corpus 
    """
    pass

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[dict]:
    """
    This functions breaks down the corpus into chunks with given size. 

    Args:
        text: corpus
        chunk_size: size of each chunk
        overlap: size of text overlap between two chunks
    """
    pass


