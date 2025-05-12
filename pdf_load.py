import spacy
from PyPDF2 import PdfReader

def load_pdf_text_to_spacy(pdf_path):
    # Load the spaCy language model
    nlp = spacy.load("en_core_web_sm")
    
    # Read the PDF file
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    # Process the text with spaCy
    doc = nlp(text)
    return doc

# Example usage
if __name__ == "__main__":
    pdf_path = "GD_HMI Institute of Health Science_081024.pdf"  # Replace with your PDF file path
    doc = load_pdf_text_to_spacy(pdf_path)
    
    # Print sentences from the processed document
    for sentence in doc.sents:
        print(sentence)