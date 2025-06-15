import os
import io
import json
import re
from PIL import Image
from pdf2image import convert_from_path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google Cloud Imports
from vertexai.vision_models import Image as GeminiImage
from vertexai.preview.generative_models import GenerativeModel, Part,Image # Ensure Part is imported
import vertexai

# PDF Generation Imports
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Spacer, SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# --- Configuration ---
# Get configuration from environment variables
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT_ID')

if not PROJECT_ID:
    raise ValueError("Missing required environment variable GOOGLE_CLOUD_PROJECT_ID")

# Initialize Vertex AI for Gemini with the specified model
vertexai.init(project=PROJECT_ID, location="us-central1") # 'us-central1' is common for Gemini models
GEMINI_MODEL = GenerativeModel("gemini-2.5-flash-preview-05-20") # Using the exact model requested

def natural_sort_key(s):
    """
    Helper function to sort strings containing numbers in a natural way.
    For example: ['1', '2', '10'] instead of ['1', '10', '2']
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]

# --- Utility Functions ---
def create_directory_if_not_exists(path):
    """Creates a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    print("✓ Directory structure verified")

def get_mime_type(image_path):
    """Determines MIME type based on image file extension."""
    if image_path.lower().endswith(".png"):
        return "image/png"
    elif image_path.lower().endswith((".jpeg", ".jpg")):
        return "image/jpeg"
    else:
        raise ValueError(f"Unsupported image format: {image_path}. Only PNG and JPEG are supported.")

def convert_pdf_to_images(pdf_path, output_folder):
    """
    Converts a PDF file into a series of images, one per page.
    DPI is set to 300 for better quality for Gemini Vision.
    """
    create_directory_if_not_exists(output_folder)
    print("📄 Converting PDF to images...")
    try:
        images = convert_from_path(pdf_path, dpi=300)
        page_image_paths = []
        for i, img in enumerate(images):
            page_image_path = os.path.join(output_folder, f"page_{i+1}.png")
            img.save(page_image_path, "PNG")
            page_image_paths.append(page_image_path)
            print(f"  ✓ Page {i+1} processed")
        print(f"✓ PDF conversion complete: {len(page_image_paths)} pages processed")
        return page_image_paths
    except Exception as e:
        print("❌ Error: PDF conversion failed")
        print("  Please ensure Poppler is correctly installed on your system")
        return []

def _strip_markdown_json(text):
    """
    Strips markdown code block fences (```json) from a string using regex,
    finding the first JSON block and ignoring surrounding text.
    """
    # Regex to find the first occurrence of ```json...```
    # The `.*?` ensures a non-greedy match, and `re.DOTALL` allows . to match newlines.
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # If no markdown block is found, assume the entire text is JSON and try to parse it.
    # This might happen if Gemini doesn't wrap the JSON as expected.
    return text.strip()

# --- Part 1: Initial Processing & Student Info Extraction ---
def extract_student_info(answer_script_page1_path):
    """
    Extracts student information from the first page of the answer script.
    """
    print("\n📋 Extracting student information...")

    with open(answer_script_page1_path, "rb") as f:
        image_bytes = f.read()
    mime_type = get_mime_type(answer_script_page1_path)

    prompt = f"""
    Extract the following student information from the provided image of a student's answer script cover page.
    If a piece of information is not found, use "N/A".
    Return the information in a JSON object with keys: "name", "enrollment_number", "semester", "course_name", "date".
    Example:
    {{"name": "John Doe", "enrollment_number": "12345", "semester": "Fall 2023", "course_name": "Computer Science I", "date": "2023-12-15"}}
    """

    try:
        response = GEMINI_MODEL.generate_content(
            [Part.from_text(prompt), Part.from_data(data=image_bytes, mime_type=mime_type)],
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        gemini_response_text = _strip_markdown_json(response.text)
        student_info = json.loads(gemini_response_text)
        print("✓ Student information extracted successfully")
        return student_info
    except json.JSONDecodeError as e:
        print("❌ Error: Could not parse student information")
        return {"name": "N/A", "enrollment_number": "N/A", "semester": "N/A", "course_name": "N/A", "date": "N/A"}
    except Exception as e:
        print("❌ Error: Failed to extract student information")
        return {"name": "N/A", "enrollment_number": "N/A", "semester": "N/A", "course_name": "N/A", "date": "N/A"}

# --- Part 2: Question Paper Processing ---
def extract_questions_from_paper(qp_image_paths):
    """
    Extracts questions and their marks from the question paper images.
    """
    print("\n📝 Analyzing question paper...")
    questions_data = []

    for i, qp_path in enumerate(qp_image_paths):
        print(f"  Processing page {i+1} of question paper...")
        with open(qp_path, "rb") as f:
            image_bytes = f.read()
        mime_type = get_mime_type(qp_path)

        prompt = f"""
        Analyze the provided image, which is a page from a question paper.
        Extract all questions and their associated marks.
        Questions can be numbered like '1', '2', or '1a', '1b', '2a', or '1(i)','1(ii)' etc.
        If a question has sub-parts like 'a.', 'b.', 'c.' under a main number like '1', combine them into '1a', '1b', '1c'.
        If marks are mentioned (e.g., (5 marks), [10]), extract them as an integer. If not, use 0.
        Represent mathematical formulas using standard plain text notation (e.g., E=mc^2, x^2 + y/z, sqrt(x)).
        Return a JSON array of objects, each with "question_number", "question_text", and "max_marks".
        Example:
        [
          {{"question_number": "1a", "question_text": "What are the differences between X and Y? Explain in detail.", "max_marks": 5}},
          {{"question_number": "1b", "question_text": "Describe the Z algorithm with steps.", "max_marks": 10}},
          {{"question_number": "2", "question_text": "Derive the equation for force $F = ma$.", "max_marks": 8}}
        ]
        """

        try:
            response = GEMINI_MODEL.generate_content(
                [Part.from_text(prompt), Part.from_data(data=image_bytes, mime_type=mime_type)],
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            gemini_response_text = _strip_markdown_json(response.text)
            extracted_qps = json.loads(gemini_response_text)
            for q in extracted_qps:
                if 'max_marks' in q:
                    try:
                        q['max_marks'] = int(q['max_marks'])
                    except (ValueError, TypeError):
                        q['max_marks'] = 0
                else:
                    q['max_marks'] = 0
                questions_data.append(q)
            print(f"  ✓ Page {i+1} processed successfully")
        except json.JSONDecodeError as e:
            print(f"  ❌ Error: Could not parse questions from page {i+1}")
        except Exception as e:
            print(f"  ❌ Error: Failed to process page {i+1}")

    questions_data.sort(key=lambda x: natural_sort_key(x['question_number']))
    print(f"✓ Question paper analysis complete: {len(questions_data)} questions extracted")
    return questions_data

# --- Part 3: Answer Script Content Extraction & Question Pairing (Unified with Gemini Vision) ---
def extract_student_answers_from_script(answer_script_image_paths, question_paper_data):
    """
    Extracts student answers for all questions from the answer script.
    """
    print(f"\n📖 Analyzing student's answer script...")
    print(f"  Processing {len(answer_script_image_paths)} pages...")

    # Prepare all answer script images as Gemini Part objects using Part.from_data
    answer_script_parts = []
    for p_path in answer_script_image_paths:
        with open(p_path, "rb") as f:
            image_bytes = f.read()
        mime_type = get_mime_type(p_path)
        answer_script_parts.append(Part.from_data(data=image_bytes, mime_type=mime_type))

    # Compile all questions into a clear string for the prompt
    # Providing the question paper helps Gemini map student answers correctly, even if they don't exactly match
    questions_list_str = "\n".join([
        f"- Question {q['question_number']}: {q['question_text']} (Max Marks: {q['max_marks']})"
        for q in question_paper_data
    ])

    # Master prompt for extracting all answers from the entire script
    master_prompt_parts = [
        Part.from_text(f"""
        You are an expert examiner and content extraction specialist.
        Analyze the provided set of images, which collectively represent a student's complete answer script.
        **Important:** Student answers may appear in any order throughout the script, not necessarily following the question paper's sequence. Also, parts of a single question (e.g., 1a, 1b) or even the entire answer for one question may span across multiple pages. Ensure you capture ALL content related to a specific question, regardless of its position in the script.
        Your primary goal is to **accurately extract the student's full written answer for each question from the question paper, identifying it strictly by its question number.**
        **DO NOT evaluate, compare, or summarize the correctness of the answer at this stage.** Simply transcribe what the student has written, *associated with the correct question number*.

        Identify the question number **strictly from the provided `questions_list_str`**. The student may identify their answer in several ways:
        - Explicitly: 'Ans-1a', 'Q1', '1.', '2b', 'Question 3'.
        - **Implicitly: By writing the full question text (or a significant part of it) before their answer. In such cases, match this text to the questions in the `questions_list_str`.**
        - **Directly: By starting an answer without an explicit question number, but the content strongly matches a known question.**
        - **Contextually (for sub-parts): If a student writes only 'b)', 'c)', or similar (e.g., 'ii', 'III') after an answer to a preceding main question or sub-part (e.g., 'Q1 a)'), **and no new main question number is visible**, assume it refers to the next sequential sub-part of the *immediately preceding main question*. Infer the full question number (e.g., '1b', '1c').

        **Crucial Instruction: For any question, if a student has crossed out or clearly indicated an answer as 'rough work' or 'cancelled', ignore that content.** Focus on and extract only the final, legible, and non-crossed-out answer provided for that question. If there are multiple non-crossed-out attempts, extract the most comprehensive or clearly intended final one.
        For each identified question, extract the complete text of the student's answer.
        This includes:
        - All handwritten text.
        - **Any relevant diagrams (describe their key visual elements briefly, even if not explicitly labeled by the student, if they logically contribute to the answer).** For example: "[DIAGRAM: A hand-drawn diagram of a CRT monitor showing electron gun, deflection plates, and screen.]"
        - **Any tabular data (describe its content or structure, or try to extract into a simple row-by-row format if possible, even if not explicitly drawn as a formal table).** For example: "[TABLE: A comparison table with columns 'Feature' and 'Description' and several rows of data.]"
        - Any code snippets (transcribe accurately).
        - **If there are any other visual elements, describe them briefly.** For example: "[IMAGE: A hand-drawn graph showing a linear relationship between variables X and Y.]"
        - **If there are spelling or grammatical errors,make them correct but do not change the meaning of the answer.**
        - **Any mathematical formulas or derivations (transcribe using standard math-friendly Unicode symbols when possible):**
        - Use superscripts for powers (e.g., x², a³)
            - Use subscripts for molecules or indices (e.g., H₂O, A₁)
            - Use √x instead of sqrt(x)
            - Use fraction-style slash (/) instead of `/` when applicable (e.g., y/z)
        **Ignore any content that appears to be rough work, scratchpad calculations, or unrelated doodles, especially if it's in margins, heavily crossed out, or clearly separated from main answers.** Your focus is on extracting the student's final submitted answers.
        If a question from the question paper (listed below) is not explicitly answered by the student, indicate "Not Attempted".
        Ensure that the extracted answer for each question is complete, even if it spans multiple pages.

        Here are the questions from the original question paper to guide your identification:
        ```
        {questions_list_str}
        ```

        Provide the extracted answers as a JSON array. Each object in the array should correspond to a question from the `question_paper_data` list.
        Use the exact 'question_number' from the `question_paper_data` provided above.
        
        **CRITICAL INSTRUCTION**: Your response MUST be a perfectly valid JSON object.

        **VERY IMPORTANT ESCAPING RULES FOR JSON STRINGS**:
        -Any literal double quote (") within a string value MUST be escaped as \\".
        -Any literal backslash (\) within a string value MUST be escaped as \\\\.
        -Newlines within strings should be \\n.
        DO NOT include any explanatory text or markdown outside the JSON object.

        Structure of each answer object:
        {{
          "question_number": "...", // The question number from the question paper
          "student_written_answer_content": "...", // The student's full answer, including descriptions of visuals, tables, formulas. Use "Not Attempted" if no answer. **If specific words or phrases are completely illegible, use '[ILLEGIBLE]' in their place.**
          "attempted": true/false // Set to false if no discernible answer was found for this question
        }}

        Example of a JSON array response:
        ```json
        [
          {{
            "question_number": "1a",
            "student_written_answer_content": "Screen Aspect Ratio: It is ratio width to length of the screen or ratio of number of pixels in row to total of column. Resolution: It is the product of number of pixels in row and column in the display screen. Suppose a screen having 1240 pixels per row and 570 pixels per column. Screen aspect ratio = $1240 / 570 = 124 / 57$. Resolution = $1240 * 570$.",
            "attempted": true
          }},
          {{
            "question_number": "1b",
            "student_written_answer_content": "[DIAGRAM: A hand-drawn diagram of a Cathode Ray Tube (CRT) monitor, showing internal components. Labeled parts include 'focus system', 'horizontal plate', 'Phosphor coated screen', 'Connector Pin', 'Base', 'Vertical shift plate'.]\n\nOperations:\n1. Electron gun emits electron beam\n2. Intensity of electron beam is adjusted by voltage controller to produce desired colour brightness.\n3. Focus system focus beam to produce a sharp image.\n4. Vertical and horizontal plates deviate electron beam to desired portion of screen.\n5. When electron beam strikes phosphor coated screen, it illuminates producing color.",
            "attempted": true
          }},
          {{
            "question_number": "1c",
            "student_written_answer_content": "Difference b/w boundary fill algorithm and flood fill algorithm:\nBoundary fill algorithm:\n  input: Boundary pixel color, specified color, seed pixel\n  condition: If connected pixel color is not equal to boundary color and also not equal to specified color then push it to stack.\nFlood-Fill algorithm:\n  input: Specified color, interior color, seed pixel\n  condition: If connected pixel color is  equal to interior color push it to stack.",
            "attempted": true
          }},
          {{
            "question_number": "1d",
            "student_written_answer_content": "If we are given some points then a smallest convex polygon which consist of all the points is known as convex hull. Major objective of using convex hull is to scale image or to clipping of graphics.",
            "attempted": true
          }},
          {{
            "question_number": "1e",
            "student_written_answer_content": "It is the buffer memory which stores graphic definition. Ex: It is used in Random Scan and Vector devices. [DIAGRAM: A block diagram showing connections between various computer components and a CRT monitor, including CPU, I/P Port, Device Controller, Keyboard, Mouse, and Frame Buffer].",
            "attempted": true
          }},
          {{
            "question_number": "2", // Example for a question not answered
            "student_written_answer_content": "Not Attempted",
            "attempted": false
          }}
        ]
        ```
        """)
    ] + answer_script_parts # Add all image parts after the text prompt

    student_answers_raw_data = []

    try:
        response = GEMINI_MODEL.generate_content(
            master_prompt_parts,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        gemini_output = _strip_markdown_json(response.text)
        extracted_answers_list = json.loads(gemini_output)

        for qp_q in question_paper_data:
            q_num = qp_q["question_number"]
            matched_answer = next((ans for ans in extracted_answers_list if ans.get("question_number") == q_num), None)

            student_answer_raw = ""
            attempted = False
            if matched_answer and matched_answer.get("attempted", False) and matched_answer.get("student_written_answer_content") and matched_answer["student_written_answer_content"].strip().upper() != "NOT ATTEMPTED":
                student_answer_raw = matched_answer["student_written_answer_content"].strip()
                attempted = True
                print(f"  ✓ Answer extracted for Question {q_num}")
            else:
                print(f"  ⚠️ No answer found for Question {q_num}")
                student_answer_raw = "Not Attempted" # Explicitly set for clearer handling later

            student_answers_raw_data.append({
                "question_number": q_num,
                "question_text": qp_q["question_text"],
                "total_marks": qp_q["max_marks"], # Use max_marks from QP
                "student_answer_raw": student_answer_raw,
                "attempted": attempted,
                "evaluation": {
                    "correct_answer": "N/A", "mistakes": [], "feedback": "N/A", "assigned_marks": 0
                }
            })
    except json.JSONDecodeError as e:
        print("❌ Error: Could not parse student answers")
        print("  Please try again or contact support if the issue persists")
    except Exception as e:
        print("❌ Error: Failed to analyze answer script")
        print("  Please try again or contact support if the issue persists")

    print("✓ Answer script analysis complete")
    return student_answers_raw_data

# --- Part 4: Evaluation Logic ---
def generate_correct_answer(question_text):
    """
    Generates a concise correct answer for a given question using Gemini.
    """
    prompt = f"""
    You are an expert academic content writer. Generate a factually accurate, well-structured answer for the question below.

    ✦ Keep the answer **balanced**: not too short, not too long (ideally 4-6 bullet points or 4-6 lines).
    ✦ Focus on **key points** and **clarity**.
    ✦ Avoid lengthy introductions, repetition, or unnecessary elaboration.
    ✦ Format the answer in **bullet points** if possible (except for code, tables, or formulas).
    ✦ For diagrams: Briefly describe what should be shown.
    ✦ For tables: Describe the structure (rows/columns and sample values).
    ✦ For code: Give a minimal, correct snippet only.
    
    ✦ ✨ VERY IMPORTANT: Use proper **math formatting**:
    - Use **Unicode superscripts** for powers (e.g., x², a³, m⁻¹)
    - Use **Unicode subscripts** for chemical/molecular terms (e.g., H₂O, CO₂)
    - Use **√x** instead of `sqrt(x)`
    - Use **fraction-style slash (/)** when appropriate (e.g., y/z instead of y/z)

    DO NOT include any extra comments or conversational text—only the answer content.

    Question: {question_text}
    """
    print(f"--- Generating correct answer for: {question_text[:70]}... ---")
    response = GEMINI_MODEL.generate_content(
        [Part.from_text(prompt)],
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    return response.text

def evaluate_answer(question_data):
    """
    Evaluates a student's answer against the correct answer.
    """
    question_text = question_data["question_text"]
    student_answer_raw = question_data["student_answer_raw"]
    total_marks = question_data["total_marks"]

    print(f"\n📊 Evaluating Question {question_data['question_number']}...")
    
    # Generate correct answer
    correct_answer = generate_correct_answer(question_text)
    question_data["evaluation"]["correct_answer"] = correct_answer

    if not question_data["attempted"] or student_answer_raw.strip().upper() == "NOT ATTEMPTED":
        question_data["evaluation"]["mistakes"] = ["Question not attempted"]
        question_data["evaluation"]["feedback"] = "No answer provided"
        question_data["evaluation"]["assigned_marks"] = 0
        print(f"  ⚠️ Question not attempted")
        return question_data

    try:
        prompt = f"""
       You are a kind, experienced teacher grading student answers. Be encouraging, fair, and student-friendly.
    
        Evaluate the student's answer based on the given question and the ideal answer. Reward effort generously and do not expect perfect model-level answers from the student. If the student captures the main idea or makes a good attempt, assign decent marks. Avoid being harsh for minor mistakes.
        **When referring to mathematical expressions, use standard Unicode math symbols wherever possible (e.g., x², √x, a/b, H₂O) instead of plain text like x^2 or sqrt(x).**

                
        Question: {question_text}
        Total Marks for this question: {total_marks}
        
        Correct Answer:
        ```
        {correct_answer}
        ```
        
        Student's Answer:
        ```
        {student_answer_raw}
        ```
        
        Based on the above, provide:
        1. **mistakes**: A short list of any key missing or incorrect points, if any. Keep it short. If the student mostly got it right, write: "No major mistakes."
        2. **feedback**: Give short, encouraging advice (1-2 lines) to help the student improve. Avoid lengthy explanations.
        3. **assigned_marks**: Give a fair mark out of {total_marks}, with a positive bias. Even a partial but relevant answer should get more than 0. Use integers or one decimal place if needed.

        
        **CRITICAL INSTRUCTION**: Your response MUST be a perfectly valid JSON object.

        **VERY IMPORTANT ESCAPING RULES FOR JSON STRINGS**:
        -Any literal double quote (") within a string value MUST be escaped as \\".
        -Any literal backslash (\) within a string value MUST be escaped as \\\\.
        -Newlines within strings should be \\n.
        DO NOT include any explanatory text or markdown outside the JSON object.

        Format your response as a JSON object with keys: "mistakes", "feedback", "assigned_marks".
        Example:
        {{
            "mistakes": ["- Missing definition of X...", "- Incorrect formula for Y (E=mc^3 instead of E=mc^2)."],
            "feedback": "Focus on accurate definitions...",
            "assigned_marks": 3.5
        }}
        """
        response = GEMINI_MODEL.generate_content(
            [Part.from_text(prompt)],
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, # Typo fix: BLOCK_NONE
            }
        )
        gemini_response_text = _strip_markdown_json(response.text)
        evaluation_result = json.loads(gemini_response_text)

        assigned_marks = evaluation_result.get("assigned_marks", 0)
        # Ensure assigned_marks is a number and within bounds
        try:
            assigned_marks = float(assigned_marks) # Allow float marks
        except (ValueError, TypeError):
            assigned_marks = 0

        if isinstance(total_marks, (int, float)):
            assigned_marks = min(max(0.0, assigned_marks), float(total_marks))
        else: # Fallback if total_marks is not a number
            assigned_marks = min(max(0.0, assigned_marks), 10.0) # Default to 10 max if type unknown

        question_data["evaluation"]["mistakes"] = evaluation_result.get("mistakes", ["N/A"])
        # Ensure mistakes is a list
        if not isinstance(question_data["evaluation"]["mistakes"], list):
            question_data["evaluation"]["mistakes"] = [str(question_data["evaluation"]["mistakes"])]

        question_data["evaluation"]["feedback"] = evaluation_result.get("feedback", "N/A")
        question_data["evaluation"]["assigned_marks"] = assigned_marks

        print(f"  ✓ Evaluation complete: {assigned_marks}/{total_marks} marks")
        return question_data
    except json.JSONDecodeError as e:
        print(f"  ❌ Error: Could not evaluate answer")
        return question_data
    except Exception as e:
        print(f"  ❌ Error: Evaluation failed")
        return question_data

def generate_evaluated_pdf(original_qp_path, evaluated_answers, student_info, output_pdf_path):
    """
    Generates the final evaluated PDF report.
    """
    print("\n📑 Generating evaluation report...")

    create_directory_if_not_exists(os.path.dirname(output_pdf_path))

    doc = SimpleDocTemplate(output_pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("<b>Student Answer Script Evaluation Report</b>", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))

    # Student Info
    story.append(Paragraph("<b>Student Information:</b>", styles['h2']))
    story.append(Paragraph(f"<b>Name:</b> {student_info.get('name', 'N/A')}", styles['Normal']))
    story.append(Paragraph(f"<b>Enrollment No.:</b> {student_info.get('enrollment_number', 'N/A')}", styles['Normal']))
    story.append(Paragraph(f"<b>Semester:</b> {student_info.get('semester', 'N/A')}", styles['Normal']))
    story.append(Paragraph(f"<b>Course:</b> {student_info.get('course_name', 'N/A')}", styles['Normal']))
    story.append(Paragraph(f"<b>Date:</b> {student_info.get('date', 'N/A')}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Total Marks
    total_marks_obtained = sum(ans["evaluation"]["assigned_marks"] for ans in evaluated_answers if isinstance(ans["evaluation"]["assigned_marks"], (int, float)))
    total_possible_marks = sum(ans["total_marks"] for ans in evaluated_answers if isinstance(ans["total_marks"], (int, float)))

    story.append(Paragraph(f"<b>Total Marks Obtained:</b> {total_marks_obtained} / {total_possible_marks}", styles['h2']))
    story.append(Spacer(1, 0.5 * inch))

    # Add each question's evaluation
    for ans in evaluated_answers:
        # Ensure all HTML tags are properly closed
        question_text = ans.get('question_text', '').replace('\n', '<br/>')
        story.append(Paragraph(f"<font color='blue'><b>Question {ans['question_number']}:</b></font> {question_text}", styles['h3']))
        story.append(Paragraph(f"<b>Marks:</b> {ans['evaluation']['assigned_marks']} / {ans['total_marks']}", styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))

        story.append(Paragraph("<b>Student's Answer (Extracted):</b>", styles['h4']))
        # Clean the student answer text
        student_answer = ans.get('student_answer_raw', '').replace('\n', '<br/>')
        story.append(Paragraph(f"<font name='Helvetica'>{student_answer}</font>", styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))

        story.append(Paragraph("<b>Correct Answer:</b>", styles['h4']))
        correct_answer = ans['evaluation'].get('correct_answer', '').replace('\n', '<br/>')
        story.append(Paragraph(correct_answer, styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))

        story.append(Paragraph("<b>Mistakes:</b>", styles['h4']))
        if isinstance(ans['evaluation']['mistakes'], list):
            for mistake in ans['evaluation']['mistakes']:
                clean_mistake = str(mistake).replace('\n', '<br/>')
                story.append(Paragraph(f"- {clean_mistake}", styles['Normal']))
        else: # Fallback if for some reason it's not a list
            clean_mistake = str(ans['evaluation']['mistakes']).replace('\n', '<br/>')
            story.append(Paragraph(clean_mistake, styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))

        story.append(Paragraph("<b>Feedback:</b>", styles['h4']))
        feedback = ans['evaluation'].get('feedback', '').replace('\n', '<br/>')
        story.append(Paragraph(feedback, styles['Normal']))
        story.append(Spacer(1, 0.3 * inch))

    try:
        doc.build(story)
        print("✓ Evaluation report generated successfully")
    except Exception as e:
        print("❌ Error: Failed to generate evaluation report")

# Google Cloud Imports
from vertexai.generative_models import GenerativeModel, Part, HarmCategory, HarmBlockThreshold
# --- Main Orchestration Function ---
def main_evaluation_pipeline(answer_script_pdf_path, question_paper_pdf_path):
    """
    Main pipeline for evaluating student answer scripts.
    """
    print("\n🚀 Starting Answer Script Evaluation")
    print("----------------------------------------")

    create_directory_if_not_exists("output_images")
    create_directory_if_not_exists("block_images") # Not used, but kept for consistency
    create_directory_if_not_exists("evaluated_pdfs")

    # --- Part 1: Initial Processing & Student Info Extraction ---
    answer_script_images_output_folder = os.path.join("output_images", "answer_script")
    answer_script_image_paths = convert_pdf_to_images(answer_script_pdf_path, answer_script_images_output_folder)
    if not answer_script_image_paths:
        print("Pipeline aborted: No images generated from answer script PDF.")
        return

    student_info = extract_student_info(answer_script_image_paths[0])

    # --- Part 2: Question Paper Processing ---
    question_paper_images_output_folder = os.path.join("output_images", "question_paper")
    question_paper_image_paths = convert_pdf_to_images(question_paper_pdf_path, question_paper_images_output_folder)
    if not question_paper_image_paths:
        print("Pipeline aborted: No images generated from question paper PDF.")
        return

    question_paper_data = extract_questions_from_paper(question_paper_image_paths)
    if not question_paper_data:
        print("Pipeline aborted: No questions extracted from question paper.")
        return

    # --- Part 3: Answer Script Content Extraction & Question Pairing ---
    # Pass all answer script pages to extract answers for all questions
    evaluated_answers_data = extract_student_answers_from_script(answer_script_image_paths, question_paper_data)

    # --- Part 4: Evaluation Logic ---
    print("\n--- Starting Answer Evaluation ---")
    for i, ans in enumerate(evaluated_answers_data):
        print(f"  Evaluating Q{ans['question_number']} ({i+1}/{len(evaluated_answers_data)})")
        evaluated_answers_data[i] = evaluate_answer(ans)
    print("--- Answer Evaluation Complete ---")

    # --- Part 5: Final PDF Generation ---
    # Sanitize enrollment_no for filename
    enrollment_no = student_info.get('enrollment_number', 'unknown_student').replace('/', '_').replace('\\', '_').replace(' ', '_')
    output_filename = f"evaluated_script_{enrollment_no}.pdf"
    output_pdf_path = os.path.join("evaluated_pdfs", output_filename)
    generate_evaluated_pdf(question_paper_pdf_path, evaluated_answers_data, student_info, output_pdf_path)

    print("\n✨ Evaluation Pipeline Complete!")
    print("----------------------------------------")
    print("You can now download the evaluation report")

# --- To run the script ---
if __name__ == "__main__":
    # Define your input PDF paths
    student_ans_script_path = "storage/answer_Test_c7_merged.pdf"
    question_paper_path = "storage/Question_test_c7.pdf"

    # Check if input PDFs exist
    if not os.path.exists(student_ans_script_path):
        print(f"Error: Student answer script PDF '{student_ans_script_path}' not found.")
        print("Please ensure the answer script PDF is in the server/storage directory.")
    elif not os.path.exists(question_paper_path):
        print(f"Error: Question paper PDF '{question_paper_path}' not found.")
        print("Please ensure the question paper PDF is in the server/storage directory.")
        print("This file is crucial for Gemini to extract questions and marks.")
    else:
        main_evaluation_pipeline(student_ans_script_path, question_paper_path)

