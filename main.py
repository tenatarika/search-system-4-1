import uvicorn
from fastapi import FastAPI, UploadFile, File
from nltk.corpus import stopwords

from engine import SearchEngine
from textRank import TextRank4Sentences
from vacabulary_detector import NewLangDetector
from rake_nltk import Rake
app = FastAPI()


def remover_stop_words(text: str):
    return [word for word in text if not word in stopwords.words("russian")]


@app.post("/stopwords")
def remove_stopwords(text: str):
    return {"result": "".join(remover_stop_words(text))}


@app.get("/files")
def get_files():
    engine = SearchEngine()
    files = engine.get_files()
    return {"files": files}


@app.post("/upload-file/")
async def create_upload_file(uploaded_file: UploadFile = File(...)):
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())
    return {"info": f"file '{uploaded_file.filename}' saved at '{file_location}'"}


@app.get("/unic_for_each_file")
def get_unic_words():
    engine = SearchEngine()
    unic_words = engine.get_unic_words_for_each_file()
    return {"file_name - unic_words": unic_words}


@app.get("/get_tf")
def get_suggestion_files(query: str):
    engine = SearchEngine()
    engine.calculate_res("".join(remover_stop_words(query)))
    return engine.files


@app.get("/detect_language")
def get_language_detection(file_path: str):
    service = NewLangDetector()
    return service.get_language(file_path)


@app.get("/get_referators")
def get_referators(file_path: str):
    with open(file_path) as file:
        text = file.read()

    tr4sh = TextRank4Sentences()
    tr4sh.analyze(text)
    tr4sh_analize = tr4sh.get_top_sentences(5)

    r = Rake()
    r.extract_keywords_from_text(text)
    r.get_ranked_phrases()
    return {"first": tr4sh.get_top_sentences(5), "next": r.get_ranked_phrases()}


if __name__ == '__main__':
    uvicorn.run(
        app,
        host="localhost",
        port=8888,
    )


