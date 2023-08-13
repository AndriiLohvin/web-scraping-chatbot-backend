import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Form
from app.Utils.pinecone import get_answer, get_context, train_csv, train_pdf, train_txt, train_ms_word, delete_all_data, set_prompt, delete_data_by_metadata
from fastapi.responses import StreamingResponse
import os

router = APIRouter()


# @router.post("/addCSVFiles")
# def addCSVFiles(file: UploadFile = File(...)):
#     if file.filename.endswith(".csv") == False:
#       raise HTTPException(
#         status_code=500, detail="Only support csv file")
#     destination_directory = "./app/training-files/"
#     destination_file_path = os.path.join(destination_directory, file.filename)
#     os.makedirs(destination_directory, exist_ok=True)
#     with open(destination_file_path, "wb") as destination_file:
#         shutil.copyfileobj(file.file, destination_file)
#     train_file(file.filename)

supported_file_extensions = [".csv", ".pdf", ".txt", ".doc", ".docx"]


@router.post("/add-training-file")
def add_training_file_api(file: UploadFile = File(...)):
    extension = os.path.splitext(file.filename)[1]
    if extension not in supported_file_extensions:
        raise HTTPException(
            status_code=500, detail="Only support csv file")
    # print("valid filetype")
    try:
        # save file to server
        directory = "./train-data"
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(f"{directory}/{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # add training file
        if extension == ".csv":
            train_csv(file.filename)
        elif extension == ".pdf":
            train_pdf(file.filename)
        elif extension == ".txt":
            train_txt(file.filename)
        elif extension == ".docx":
            train_ms_word(file.filename)
        print("end-training")
        # add_file(file.filename)
    except Exception as e:
        print("training error")
        raise HTTPException(
            status_code=500, detail=e)


@router.post("/similar-context")
def find_similar_context(msg: str = Form(...)):
    print("msg: " + str(msg))
    if len(msg.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No Input Message",
        )
    return get_context(msg)


@router.post("/user-question")
def answer_to_user_question(msg: str = Form(...)):
    try:
        return StreamingResponse(get_answer(msg), media_type='text/event-stream')
    except Exception as e:
        raise e


@router.post("/clear-database")
def clear_database():
    delete_all_data()
    return True

@router.post("/clear-database-by-metadata")
def clear_database_by_metadata(filename: str = Form(...)):
    delete_data_by_metadata(filename)

@router.post("/set-prompt")
def set_prompt_by_user(prompt: str = Form(...)):
    set_prompt(prompt)