# app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from langchain.chains import LLMChain
from pipelines.basic_pipeline import setup_basic_pipeline, run_basic_pipeline

# Set up logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("app")

app = FastAPI(title="Autonomous SaaS Web Application")

# Initialize LangChain pipeline
llm_chain = setup_basic_pipeline()


class ContentRequest(BaseModel):
    input_text: str


@app.post("/generate-content/")
async def generate_content(request: ContentRequest):
    logger.info(f"Received content generation request: {request.input_text}")
    try:
        content = run_basic_pipeline(llm_chain, request.input_text)
        return {"generated_content": content}
    except Exception as e:
        logger.error(f"Error generating content: {e}")
        raise HTTPException(status_code=500, detail="Content generation failed.")


@app.get("/")
async def root():
    return {"message": "Welcome to the Autonomous SaaS Web Application"}
