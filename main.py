import uvicorn
from fastapi import FastAPI

from engine import SearchEngine

app = FastAPI()


@app.get("/files")
def get_files():
    engine = SearchEngine()
    files = engine.get_files()
    return {"files": files}


@app.get("/unic_for_each_file")
def get_unic_words():
    engine = SearchEngine()
    unic_words = engine.get_unic_words_for_each_file()
    return {"file_name - unic_words": unic_words}


@app.get("/get_tf")
def get_suggestion_files(query: str):
    engine = SearchEngine()
    engine.calculate_res(query)
    return engine.files


if __name__ == '__main__':
    uvicorn.run(
        app,
        host="localhost",
        port=8888,
    )


