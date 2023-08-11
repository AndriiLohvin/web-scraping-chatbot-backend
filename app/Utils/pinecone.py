from langchain.schema import Document
import pandas as pd
from fastapi import UploadFile, File
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader, PyPDFLoader, TextLoader, Docx2txtLoader

from dotenv import load_dotenv
import os
import pinecone
import openai
import tiktoken
import time

load_dotenv()
tokenizer = tiktoken.get_encoding('cl100k_base')

pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),  # find at app.pinecone.io
    environment=os.getenv('PINECONE_ENV'),  # next to api key in console
)

index_name = os.getenv('PINECONE_INDEX')
embeddings = OpenAIEmbeddings()
similarity_min_value = 0.5


def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


# def train_file(filename: str):
    destination_directory = "./app/training-files/"
    destination_file_path = os.path.join(destination_directory, filename)

    loader = CSVLoader(file_path=destination_file_path)
    data = loader.load()
    context = ''
    for d in data:
        context += d.page_content
    doc = Document(page_content=context, metadata={"source": filename})
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, length_function=tiktoken_len,)
    chunks = text_splitter.split_documents([doc])

    Pinecone.from_documents(
        chunks, embedding=embeddings, index_name=index_name)


def split_document(doc: Document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents([doc])
    return chunks


def train_csv(filename: str):
    start_time = time.time()
    loader = CSVLoader(file_path=f"./train-data/{filename}")
    data = loader.load()
    total_content = ""
    for d in data:
        total_content += "\n\n" + d.page_content
    doc = Document(page_content=total_content, metadata={"source": filename})
    chunks = split_document(doc)
    Pinecone.from_documents(
        chunks, embeddings, index_name=index_name)

    end_time = time.time()
    print("Elapsed time: ", end_time - start_time)
    return True


def train_pdf(filename: str):
    print("begin train_pdf")
    start_time = time.time()
    loader = PyPDFLoader(file_path=f"./train-data/{filename}")
    documents = loader.load()
    Pinecone.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name,
    )
    print("end pdf-loading")
    end_time = time.time()
    print("Elapsed time: ", end_time - start_time)
    return True


def train_txt(filename: str):
    start_time = time.time()
    loader = TextLoader(file_path=f"./train-data/{filename}")
    documents = loader.load()
    chunks = split_document(documents[0])
    Pinecone.from_documents(
        chunks, embeddings, index_name=index_name)
    end_time = time.time()
    print("Elapsed time: ", end_time - start_time)
    return True


def train_ms_word(filename: str):
    start_time = time.time()
    loader = Docx2txtLoader(file_path=f"./train-data/{filename}")
    documents = loader.load()
    chunks = split_document(documents[0])
    Pinecone.from_documents(
        chunks, embeddings, index_name=index_name)
    end_time = time.time()
    print("Elapsed time: ", end_time - start_time)


def train_text():
    print("train-begin")
    with open("./data/data.txt", "r") as file:
        content = file.read()
    doc = Document(page_content=content, metadata={"source": "data1.txt"})
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=400,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents([doc])
    Pinecone.from_documents(
        chunks, embeddings, index_name=index_name)
    print("train-end")



context = ""


def get_context(msg: str):
    print("message" + msg)
    db = Pinecone.from_existing_index(
        index_name=index_name, embedding=embeddings)
    results = db.similarity_search(msg, k=4)
    print("results-size: " + str(len(results)))
    global context
    context = ""
    for result in results:
        context += f"\n\n{result.page_content}"
    return context


def get_answer(msg: str):
    # db = Pinecone.from_existing_index(
    #     index_name=os.getenv("PINECONE_INDEX"), embedding=embeddings)
    # results = db.similarity_search(msg, k=4)
    # print("results-size: " + str(len(results)))
    # context = ""
    # for result in results:
    #     context += f"\n\n{result.page_content}"
    global context
    # print("m:", msg)
    # print(context)
    instructor = f"""
        You will act as a legal science expert.
        Please research this context deeply answer questions based on  given context as well as your knowledge.
        If you can't find accurate answer, please reply similar answer to this question or you can give related information to given questions.
        -----------------------
        This is context you can refer to.
        {context}
        -----------------------
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
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
