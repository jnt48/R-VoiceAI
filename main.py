from dotenv import load_dotenv
import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables (ensure your .env contains GOOGLE_API_KEY)
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize FastAPI app
app = FastAPI(
    title="PoliSmart Dynamic Policy Recommender",
    version="2.2",
    description="An AI-powered service to generate highly professional and personalized insurance policy recommendations based on multi-modal customer data."
)

# Allow requests from any origin (for development; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Updated Request model with additional fields for a realistic recommendation
class PolicyRecommendationRequest(BaseModel):
    customer_id: str
    insurance_type: str         # e.g., "life", "health", "auto", "home"
    customer_age: int
    employment_status: str        # e.g., "employed", "self-employed", "unemployed", "retired"
    marital_status: str           # e.g., "single", "married", "divorced", "widowed"
    dependents: int
    health_status: str            # e.g., "excellent", "good", "average", "poor"
    existing_coverage: str        # Details about any existing insurance coverage
    text_data: str                # Customer lifestyle, preferences, and long-term financial goals
    numerical_data: dict          # e.g., {"income": 50000, "existingPremium": 1200}
    additional_financial_goals: str
    behavioral_data: dict         # e.g., {"recentInteractions": "Browsed policy details, visited FAQ"}

# Response model
class PolicyRecommendationResponse(BaseModel):
    recommendation: str

def get_policy_recommendation(prompt: str) -> str:
    """
    Sends the prompt to the Gemini generative model and returns the generated recommendation.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

@app.post("/recommend_policy", response_model=PolicyRecommendationResponse)
async def recommend_policy(request: PolicyRecommendationRequest):
    """
    Generates a professional, narrative-style, and personalized insurance policy recommendation.
    Output is returned in Markdown format with a real link replacing the placeholder.
    """
    # Build a detailed prompt including extended customer data from the enhanced form.
    prompt = (
        "Please provide a highly professional and detailed personalized insurance policy recommendation based on the customer information provided below. "
        "Format your response in Markdown with clear headings, subheadings, and bullet points (avoid using numeric lists like '1.', '2.', etc.).\n\n"
        "**Customer Information**\n"
        f"- Customer ID: {request.customer_id}\n"
        f"- Insurance Type: {request.insurance_type}\n"
        f"- Customer Age: {request.customer_age}\n\n"
        "**Extended Customer Profile**\n"
        f"- Employment Status: {request.employment_status}\n"
        f"- Marital Status: {request.marital_status}\n"
        f"- Number of Dependents: {request.dependents}\n"
        f"- Health Status: {request.health_status}\n"
        f"- Existing Coverage: {request.existing_coverage}\n"
        f"- Additional Financial Goals: {request.additional_financial_goals}\n\n"
        "**Customer Profile**\n"
        f"- Lifestyle & Goals: {request.text_data}\n"
        f"- Financial Data: {request.numerical_data}\n"
        f"- Behavioral Data: {request.behavioral_data}\n\n"
        "Ensure the policy name is a real product available from our portfolio, such as 'SecureFuture Family Protector' or another verified policy name."
        "Based on the above information, produce a detailed narrative that includes a suggested policy name, recommended coverage amount, "
        "premium payment schedule, policy duration, and any additional recommendations. Also, include a link to a detailed policy document page. "
        "The final output should be easy to read, professional, and well-structured without enumerated numbering."
    )
    
    try:
        recommendation = get_policy_recommendation(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Post-process to replace any placeholder with a real URL.
    placeholder = "[Placeholder for Policy Details Page Link]"
    real_link = "https://www.exampleinsurance.com/policies/securefuture-family-protector"
    
    if placeholder in recommendation:
        recommendation = recommendation.replace(placeholder, real_link)
    else:
        recommendation += f"\n\nFor more details, please visit: [Policy Details]({real_link})"
    
    return PolicyRecommendationResponse(recommendation=recommendation)

# To run the server, use: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
