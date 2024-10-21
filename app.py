import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAI
from chromadb import Documents, EmbeddingFunction, Embeddings
import GoogleGenerativeAI as genai 
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAIEmbedding
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load.environ()

genai.configure(api_key = os.environ.get("gemini_api_key"))


def get_pdf(pdf_docs):
    text = " "

    for pdf in pdf_docs:            # loading the pdfs
        pdf_reader = PdfFileReader(pdf_docs)

        for pages in pdf:           # loading the pages
            text += pages.extract_text()

    return text


def get_pdf_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    return chunks



def setting_chromaDB(path,text_chunk,embedding):
    chroma_client = chromadb.PersistentClient(path='/home/shyan/Desktop/RAG_PIPELINE/DB')
    collection = chroma_client.create_collection(name = "Multiple_PDF_RAG")

    for i,text in enumerate(text_chunk):
        embedding = embedding.embed_text(text)
        collection.add(documents=[text], embeddings=[embedding], ids=[str(i)])


def Loadingfrom_ChromaDB(query_text):

    chroma_client = chromadb.PersistentClient(path='/home/shyan/Desktop/RAG_PIPELINE/DB')
    db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())

    return db



def get_vector_embedding(text_chunk):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    setting_chromaDB(path,text_chunk=text_chunk,embedding=embeddings)


def get_conversational_chain():
    prompt_template = ("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
    strike a friendly and converstional tone. \
    If the passage is irrelevant to the answer, you may ignore it.
    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'

    ANSWER:
    """)

    model = ChatGoogleGenerativeAIEmbedding(model = "gemini=pro",temparature = 0.3)
    prompt =PromptTemplate(template = prompt_template,input_variables = ["QUESTION","PASSAGE"])

    chain = load_qa_chain(model,chain_type="stuff",prompt = prompt)
    return chain

def user_input(user_question):
    # Basically user er query embedded kore oita ami search e pathaisi db te and db er magical algo
    # oita somehow similar dekhe return kore dise

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    embedded_query = embeddings.embed_text(user_question)

    db = Loadingfrom_ChromaDB(user_question)
    docs = db.query(query_embedding = embedded_query,n_results = 10)

    chain = get_conversational_chain()

    response = chain(
        {
            "input_documents" : docs , 
            "question" : user_question},
            return_only_outputs=True
    )
    
    return response["output_text"]
    
        
    

    #return docs

