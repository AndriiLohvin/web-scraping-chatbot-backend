from langchain.schema import Document
import pandas as pd
from fastapi import UploadFile, File
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader, PyPDFLoader, TextLoader, Docx2txtLoader
from app.Utils.web_scraping import extract_content_from_url
from app.Models.ChatbotModel import Chatbot

from dotenv import load_dotenv
import os
import pinecone
import openai
import tiktoken
import time
# from pinecone import Index

load_dotenv()
tokenizer = tiktoken.get_encoding('cl100k_base')

api_key = os.getenv('PINECONE_API_KEY')

pinecone.init(
    api_key=api_key,  # find at app.pinecone.io
    environment=os.getenv('PINECONE_ENV'),  # next to api key in console
)

index_name = os.getenv('PINECONE_INDEX')
embeddings = OpenAIEmbeddings()
similarity_min_value = 0.5
default_prompt = """
    You will act as a legal science expert.
    Please research this context deeply answer questions based on  given context as well as your knowledge.
    If you can't find accurate answer, please reply similar answer to this question or you can give related information to given questions.
    The more you can, the more you shouldn't say you don't know or this context doesn't contain accurate answer.
    If only there is never answer related to question, kindly reply you don't know exact answer.
    Don't output too many answers.
    Below is context you can refer to.
"""
prompt = default_prompt
context = ""


def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


def delete_all_data():
    # Initialize Pinecone client
    pinecone.init(api_key=api_key, environment=os.getenv('PINECONE_ENV'))

    # # Retrieve the index
    # index = pinecone.Index(index_name="your_index_name")

    # # Delete all data from the index
    # index.delete_index()

    # # Disconnect from Pinecone
    # pinecone.init()
    # pinecone.delete_index("example-index")
    # print(pinecone.list_indexes())
    if index_name in pinecone.list_indexes():
        # Delete the index
        pinecone.delete_index(index_name)
        print("Index successfully deleted.")
    else:
        print("Index not found.")

    pinecone.create_index(
        index_name,
        dimension=1536,
        metric='cosine',
        pods=1,
        replicas=1,
        pod_type='p1.x1'
    )
    print("new: ", pinecone.list_indexes())


def split_document(doc: Document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents([doc])
    return chunks


def train_csv(filename: str, namespace: str):
    start_time = time.time()
    loader = CSVLoader(file_path=f"./train-data/{filename}")
    data = loader.load()
    total_content = ""
    for d in data:
        total_content += "\n\n" + d.page_content
    doc = Document(page_content=total_content, metadata={"source": filename})
    chunks = split_document(doc)
    Pinecone.from_documents(
        chunks, embeddings, index_name=index_name, namespace=namespace)

    end_time = time.time()
    print("Elapsed time: ", end_time - start_time)
    return True


def train_pdf(filename: str, namespace: str):
    print("begin train_pdf")
    start_time = time.time()
    loader = PyPDFLoader(file_path=f"./train-data/{filename}")
    documents = loader.load()
    # chunks = split_document(documents)
    # print(type(documents))
    total_content = ""
    for document in documents:
        total_content += "\n\n" + document.page_content
    doc = Document(page_content=total_content, metadata={"source": filename})
    chunks = split_document(doc)
    Pinecone.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        namespace=namespace
    )
    print("end pdf-loading")
    end_time = time.time()
    print("Elapsed time: ", end_time - start_time)
    return True


def train_txt(filename: str, namespace: str):
    start_time = time.time()
    loader = TextLoader(file_path=f"./train-data/{filename}")
    documents = loader.load()
    total_content = ""
    for document in documents:
        total_content += "\n\n" + document.page_content
    doc = Document(page_content=total_content, metadata={"source": filename})
    print(filename)
    chunks = split_document(doc)
    print("namespace: ", namespace)
    Pinecone.from_documents(
        chunks, embeddings, index_name=index_name, namespace=namespace)
    end_time = time.time()
    print("Elapsed time: ", end_time - start_time)
    return True


def train_ms_word(filename: str, namespace: str):
    start_time = time.time()
    loader = Docx2txtLoader(file_path=f"./train-data/{filename}")
    documents = loader.load()
    chunks = split_document(documents[0])

    Pinecone.from_documents(
        chunks, embeddings, index_name=index_name, namespace=namespace)
    end_time = time.time()
    print("Elapsed time: ", end_time - start_time)


def train_text():
    print("train-begin")
    with open("./data/data.txt", "r") as file:
        content = file.read()
    doc = Document(page_content=content, metadata={"source": "data1.txt"})
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents([doc])

    Pinecone.from_documents(
        chunks, embeddings, index_name=index_name)
    print("train-end")


def train_url(url: str, namespace: str):
    content = extract_content_from_url(url)
    doc = Document(page_content=content, metadata={"source": url})
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents([doc])
    Pinecone.from_documents(
        chunks, embeddings, index_name=index_name, namespace=namespace)


def set_prompt(new_prompt: str):
    global prompt
    prompt = new_prompt


def get_context(msg: str, namespace: str, email: str):
    print("message" + msg)
    db = Pinecone.from_existing_index(
        index_name=index_name, namespace=namespace, embedding=embeddings)
    web_db = Pinecone.from_existing_index(
        index_name=index_name, embedding=embeddings)
    results = db.similarity_search(msg, k=2)
    web_results = web_db.similarity_search(msg, k=2)
    global context
    context = ""
    for web_result in web_results:
        context += f"\n\n{web_result.page_content}"
    for result in results:
        context += f"\n\n{result.page_content}"
    print(len(context)/4)
    return context


def get_answer(msg: str, namespace: str, email: str, current_bot: Chatbot):
    global context
    global prompt
    instructor = f"""
        You should answer all questions in {current_bot.language} as long as not mentioned in below prompt.
        And your tone should be {current_bot.tone} and your writing format should be {current_bot.format} and your writing style should be {current_bot.style}.
        Your answer should contains at most {current_bot.length} words as possible as you can.
        Don't output answer of more than {current_bot.length} of words.
        
        {prompt}
        -----------------------
        {context}
    """
    try:
        response = openai.ChatCompletion.create(
            model=current_bot.model,
            max_tokens=2000,
            messages=[
                {'role': 'system', 'content': instructor},
                {'role': 'user', 'content': msg}
            ],
            stream=True
        )
        for chunk in response:
            if 'content' in chunk.choices[0].delta:
                string = chunk.choices[0].delta.content
                yield string
    except Exception as e:
        print(e)

    # print(response)
    # print(response.choices[0].message.content)


def delete_data_by_metadata(filename: str, namespace: str):
    index = pinecone.Index(index_name=index_name)
    query_response = index.delete(
        namespace=namespace,
        filter={
            "source": f"{namespace}-{filename}"
        }
    )
    print(query_response)
