from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import os
import shutil
from typing import Optional, Dict
from project import main_evaluation_pipeline
import uuid
import asyncio
import queue
import threading
import sys
from io import StringIO
import time
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:5173",  # Vite default port
    "http://localhost:5174",  # Alternative Vite port
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "https://ai-script-grader-frontend.vercel.app",  # Vercel frontend URL
    "https://grader-ai.vercel.app",  # New Vercel frontend URL
    "https://ai-script-grader.onrender.com",  # Render frontend URL (if different)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # Enable credentials
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Add middleware to log requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    logger.info(f"Headers: {request.headers}")
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        raise

# Handle Google Cloud credentials
def setup_google_credentials():
    credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    if credentials_json:
        try:
            # Parse the JSON to validate it
            credentials_data = json.loads(credentials_json)
            # Write to a file
            credentials_path = "google-credentials.json"
            with open(credentials_path, 'w') as f:
                json.dump(credentials_data, f)
            # Set the environment variable to point to our file
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            print("Successfully set up Google Cloud credentials")
        except json.JSONDecodeError as e:
            print(f"Error parsing Google Cloud credentials JSON: {e}")
        except Exception as e:
            print(f"Error setting up Google Cloud credentials: {e}")
    else:
        print("Warning: GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable not set")

# Set up credentials at startup
@app.on_event("startup")
async def startup_event():
    setup_google_credentials()

# Ensure required directories exist
os.makedirs("storage", exist_ok=True)
os.makedirs("evaluated_pdfs", exist_ok=True)
os.makedirs("output_images", exist_ok=True)

# Store for evaluation logs
evaluation_logs: Dict[str, queue.Queue] = {}

class LogCapture:
    def __init__(self, evaluation_id: str):
        self.evaluation_id = evaluation_id
        self.queue = queue.Queue()
        evaluation_logs[evaluation_id] = self.queue
        self.original_stdout = sys.stdout
        self.string_io = StringIO()

    def write(self, message):
        self.original_stdout.write(message)
        self.string_io.write(message)
        self.queue.put(message)

    def flush(self):
        self.original_stdout.flush()
        self.string_io.flush()

    def close(self):
        sys.stdout = self.original_stdout
        self.queue.put(None)  # Signal end of logs
        if self.evaluation_id in evaluation_logs:
            del evaluation_logs[self.evaluation_id]

def run_evaluation(evaluation_id: str, answer_script_path: str, question_paper_path: str):
    try:
        # Set up log capture
        log_capture = LogCapture(evaluation_id)
        sys.stdout = log_capture

        # Run the evaluation
        main_evaluation_pipeline(answer_script_path, question_paper_path)

        # Get the generated evaluated PDF path
        evaluated_pdfs = os.listdir("evaluated_pdfs")
        if evaluated_pdfs:
            latest_pdf = max(
                [os.path.join("evaluated_pdfs", f) for f in evaluated_pdfs],
                key=os.path.getctime
            )
            new_pdf_path = f"evaluated_pdfs/evaluated_{evaluation_id}.pdf"
            os.rename(latest_pdf, new_pdf_path)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
    finally:
        log_capture.close()

def cleanup_evaluation_files(evaluation_id: str):
    """
    Clean up all files associated with an evaluation after it's been downloaded.
    This includes:
    - Original PDFs from storage
    - Generated images from output_images
    - The evaluated PDF
    """
    try:
        # Clean up storage files
        answer_script_path = f"storage/answer_script_{evaluation_id}.pdf"
        question_paper_path = f"storage/question_paper_{evaluation_id}.pdf"
        for path in [answer_script_path, question_paper_path]:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Cleaned up {path}")

        # Clean up output images
        answer_script_images_dir = os.path.join("output_images", "answer_script")
        question_paper_images_dir = os.path.join("output_images", "question_paper")
        for dir_path in [answer_script_images_dir, question_paper_images_dir]:
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                logger.info(f"Cleaned up images in {dir_path}")

        # Clean up evaluated PDF
        evaluated_pdf_path = f"evaluated_pdfs/evaluated_{evaluation_id}.pdf"
        if os.path.exists(evaluated_pdf_path):
            os.remove(evaluated_pdf_path)
            logger.info(f"Cleaned up {evaluated_pdf_path}")

    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

@app.post("/upload-pdfs/")
async def upload_pdfs(
    background_tasks: BackgroundTasks,
    answer_script: UploadFile = File(...),
    question_paper: UploadFile = File(...)
):
    """
    Upload answer script and question paper PDFs and start evaluation in background.
    Returns immediately with an evaluation ID.
    """
    try:
        logger.info("Starting PDF upload process")
        evaluation_id = str(uuid.uuid4())
        
        # Save files with unique names
        answer_script_path = f"storage/answer_script_{evaluation_id}.pdf"
        question_paper_path = f"storage/question_paper_{evaluation_id}.pdf"
        
        logger.info(f"Saving files to: {answer_script_path} and {question_paper_path}")
        
        # Save the uploaded files
        with open(answer_script_path, "wb") as buffer:
            shutil.copyfileobj(answer_script.file, buffer)
        
        with open(question_paper_path, "wb") as buffer:
            shutil.copyfileobj(question_paper.file, buffer)
        
        logger.info("Files saved successfully, starting evaluation")
        
        # Start evaluation in background
        background_tasks.add_task(
            run_evaluation,
            evaluation_id,
            answer_script_path,
            question_paper_path
        )
        
        logger.info(f"Evaluation started with ID: {evaluation_id}")
        
        return {
            "status": "processing",
            "message": "Evaluation started",
            "evaluation_id": evaluation_id
        }
            
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/get-evaluated-pdf/{evaluation_id}")
async def get_evaluated_pdf(evaluation_id: str, background_tasks: BackgroundTasks):
    """
    Retrieve the evaluated PDF using the evaluation ID and schedule cleanup after download.
    """
    pdf_path = f"evaluated_pdfs/evaluated_{evaluation_id}.pdf"
    
    if not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=404,
            detail="Evaluated PDF not found. The evaluation might have failed or the ID is invalid."
        )
    
    # Schedule cleanup after the file is downloaded
    background_tasks.add_task(cleanup_evaluation_files, evaluation_id)
    
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"evaluated_script_{evaluation_id}.pdf"
    )

@app.get("/evaluation-logs/{evaluation_id}")
async def get_evaluation_logs(evaluation_id: str):
    """
    Stream evaluation logs using Server-Sent Events.
    """
    if evaluation_id not in evaluation_logs:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    async def event_generator():
        log_queue = evaluation_logs[evaluation_id]
        try:
            while True:
                try:
                    # Get log message with timeout
                    message = log_queue.get(timeout=1)
                    if message is None:  # End of logs
                        break
                    yield f"data: {message}\n\n"
                except queue.Empty:
                    # Keep connection alive
                    yield "data: \n\n"
                    await asyncio.sleep(0.1)
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
        finally:
            # Clean up if not already done
            if evaluation_id in evaluation_logs:
                del evaluation_logs[evaluation_id]

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # Only run the server directly if not running under Gunicorn
    if not os.getenv("GUNICORN_RUNNING"):
        uvicorn.run(
            "api:app",
            host="0.0.0.0",
            port=int(os.getenv("PORT", "8000")),
            reload=True
        ) 