from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# 1. Load and process PDF document
def load_pdf_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()  # Returns list of Document objects

# 2. Split text into chunks
def split_documents(pages):  # Changed parameter name
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(pages)  # Directly split Document objects
