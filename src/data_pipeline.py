from pathlib import Path
import pdfplumber
import re

def load_document(pdf_path: Path) -> str:
    """
    This function loads the documents from given path.

    Args:
        pdf_path: Relative path to the pdf file.
    """
    content = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            content = content + extracted if extracted else ""

    return content if content else ""

def clean_text(text: str) -> str:
    """
    This function removes the noise like page numbers, extra whitespaces, header, artifacts from PDF extracted corpus.

    Args:
        text: corpus 
    """
    
    text = text.replace("_","")
    text = re.sub(r' +',' ', text)

    return text

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[dict]:
    """
    This functions breaks down the corpus into chunks with given size. 

    Args:
        text: corpus
        chunk_size: size of each chunk
        overlap: size of text overlap between two chunks
    """
    pass


if __name__ == "__main__":
    data = load_document(r"data\raw\the_prevention_of_cruelty_to_animals_act_1960.pdf")
    print("before")
    print(data[:500])
    clean = clean_text(data)
    print("after")
    print(clean[:500])