from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define the request body model
class Query(BaseModel):
    query: str  # The expected input field

def TwoTower(query: str):
    # Just a placeholder function to process the query
    return [["para1"], ["para2"]]

@app.post("/api/user_input")
async def pass_user_query(query: Query):
    paragraphs = TwoTower(query.query)
    return paragraphs
