from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
import os


def create_vector_store():
   docs = []
   for pdf in os.listdir("../pdf"):
      loader = PyPDFLoader(f"../pdf/{pdf}")
      docs.extend(loader.load())

   text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=30)
   chunked_docs = text_splitter.split_documents(docs)
   
   vector_db = Chroma.from_documents(chunked_docs, embedding=HuggingFaceEmbeddings(), persist_directory="../vector_store")


if __name__ == "__main__":
   create_vector_store()