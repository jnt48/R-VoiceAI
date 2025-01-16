from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Function to get the Gemini response
def get_gemini_response(teacher_code, student_code, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([teacher_code, student_code, prompt])
    return response.text

# Define the input data model using Pydantic
class CodeEvaluationRequest(BaseModel):
    teacher_code: str
    student_code: str

# Prompt for the model
input_prompt = """
You are an experienced programming instructor tasked with evaluating a student's code submission against the correct solution.
The first input is the teacher's code, and the second input is the student's code. Your task is to:

1. Thoroughly analyze both the teacher's code and the student's code for correctness.
2. Identify and highlight any differences between the two codes, focusing on:
   - Syntax errors
   - Logical errors
   - Structural mismatches (e.g., missing functions, incorrect variable names)
   - Inefficiencies or unnecessary complexities in the student's code
3. Assign a percentage score for how similar the student's code is to the teacher's code based on:
   - Structural similarity
   - Logical correctness
   - Syntax accuracy
4. Provide a final score out of 10, considering:
   - The overall correctness of the student's code
   - The quality of the code (e.g., readability, proper use of functions, error handling)
   - The degree of deviation from the correct solution

Be strict in your evaluation. If the student's code has any issues, such as errors, missing parts, or major deviations from the teacher's code, the score should reflect that with a lower percentage and score out of 10.

Ensure the score is precise and can be a decimal value (e.g., 7.5/10, 6.2/10). Output only the score and percentage match in this format:
- Percentage match: 75.5%
- Score: 6.8/10

No other text should be included.
"""

@app.post("/evaluate/")
async def evaluate_code(request: CodeEvaluationRequest):
    teacher_code = request.teacher_code
    student_code = request.student_code

    if not teacher_code.strip() or not student_code.strip():
        raise HTTPException(status_code=400, detail="Please provide both the teacher's and the student's code.")

    # Get the response from Gemini API
    response = get_gemini_response(teacher_code, student_code, input_prompt)
    
    return {"result": response}
