import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Form
from app.Utils.pinecone import get_answer, get_context
from fastapi.responses import StreamingResponse
import os

router = APIRouter()


@router.post("/addCSVFiles")
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
