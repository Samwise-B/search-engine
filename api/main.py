from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pathlib import Path
import sys

root_repo = Path(__file__).parent.parent
sys.path.append(str(root_repo))

from api.processor import process_query

# from model.two_towers import foo

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/search")
async def search(query: str):
    results = process_query(query)
    return {"results": results, "query": query}
