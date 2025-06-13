from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
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

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure required directories exist
os.makedirs("storage", exist_ok=True)
os.makedirs("evaluated_pdfs", exist_ok=True)

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
        evaluation_id = str(uuid.uuid4())
        
        # Save files with unique names
        answer_script_path = f"storage/answer_script_{evaluation_id}.pdf"
        question_paper_path = f"storage/question_paper_{evaluation_id}.pdf"
        
        # Save the uploaded files
        with open(answer_script_path, "wb") as buffer:
            shutil.copyfileobj(answer_script.file, buffer)
        
        with open(question_paper_path, "wb") as buffer:
            shutil.copyfileobj(question_paper.file, buffer)
        
        # Start evaluation in background
        background_tasks.add_task(
            run_evaluation,
            evaluation_id,
            answer_script_path,
            question_paper_path
        )
        
        return {
            "status": "processing",
            "message": "Evaluation started",
            "evaluation_id": evaluation_id
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/get-evaluated-pdf/{evaluation_id}")
async def get_evaluated_pdf(evaluation_id: str):
    """
    Retrieve the evaluated PDF using the evaluation ID.
    """
    pdf_path = f"evaluated_pdfs/evaluated_{evaluation_id}.pdf"
    
    if not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=404,
            detail="Evaluated PDF not found. The evaluation might have failed or the ID is invalid."
        )
    
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
    uvicorn.run(app, host="0.0.0.0", port=8000) 