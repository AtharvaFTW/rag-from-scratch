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
            content = content + extracted if extracted else content

    return content

def clean_text(text: str) -> str:
    """
    This function removes the noise like page numbers, extra whitespaces, header, artifacts from PDF extracted corpus.

    Args:
        text: corpus 
    """
    
    text = text.replace("_","")
    text = re.sub(r' +',' ', text)

    return text

def chunker(text: str, chunk_size: int, overlap_size: int) -> list[dict]:
    """
    This functions breaks down the corpus into chunks with given size. 

    Args:
        text: corpus
        chunk_size: size of each chunk
        overlap: size of text overlap between two chunks
    """
    
    bracket = chunk_size * 4 # Assuming the 1 token ~ 4 characters
    overlap = overlap_size * 4 # Assuming the 1 token ~ 4 characters

    # Sliding window

    l = 0 
    r = 0
    chunk_index = 0

    res = []

    while (l + bracket) <= len(text):

        r = l + bracket
        chunk_text = text[l:r]

        chunk = {
            "text":str(chunk_text),
            "chunk_index": chunk_index,
            "source": "unknown"}
        
        res.append(chunk)
        
        l = r - overlap
        chunk_index +=1

    # Fetch whatever can't fit the window    
    if (r - overlap) < len(text): 
        chunk = {
            "text": str(text[(r - overlap): len(text)]),
            "chunk_index": chunk_index,
            "source": "unknown"
        }

        res.append(chunk)
    
    return res



if __name__ == "__main__":
    data = load_document(r"data\raw\the_prevention_of_cruelty_to_animals_act_1960.pdf")
    clean = clean_text(data)
    chunks = chunker(clean, chunk_size = 512, overlap_size = 50)
    print(f"Total chunks: {len(chunks)}")
    print(chunks[0])
    print(chunks[1])
    assert chunks[1]["text"][:50*4] == chunks[0]["text"][-50*4:]
    
    