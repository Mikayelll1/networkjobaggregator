from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import PyPDF2
import io
import os
import requests
import hashlib
import secrets
import logging
import asyncio
import sys
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

# ---------------------- ENV + LOGGING SETUP ----------------------
load_dotenv()

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# ---------------------- INIT APP ----------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://job-aggregator-demo.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- AI CLIENT ----------------------
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ---------------------- USER MODELS ----------------------
users = {}
tokens = {}

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

class UserIn(BaseModel):
    username: str
    password: str
    email: Optional[str] = None

# ---------------------- AUTH ENDPOINTS ----------------------
@app.post("/register")
async def register(user: UserIn):
    if user.username in users:
        raise HTTPException(status_code=400, detail="Username already exists")
    if user.email and any(u.get("email") == user.email for u in users.values()):
        raise HTTPException(status_code=400, detail="Email already registered")

    users[user.username] = {
        "password": hash_password(user.password),
        "email": user.email
    }
    return {"msg": "User registered"}

@app.post("/login")
async def login(user: UserIn):
    hashed_input = hash_password(user.password)
    record = users.get(user.username)

    if not record:
        for uname, u in users.items():
            if u.get("email") == user.username:
                record = u
                break

    if not record or record["password"] != hashed_input:
        raise HTTPException(status_code=401, detail="Invalid username/email or password")

    token = secrets.token_hex(16)
    tokens[token] = next((k for k, v in users.items() if v == record), None)
    return {"access_token": token}

@app.get("/profile")
async def profile(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")

    token = authorization.split(" ")[1]
    username = tokens.get(token)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = users.get(username)
    return {"username": username, "email": user.get("email")}

@app.post("/logout")
async def logout(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")
    token = authorization.split(" ")[1]
    if token not in tokens:
        raise HTTPException(status_code=401, detail="Invalid token")

    del tokens[token]
    return {"msg": "Logged out successfully"}

# ---------------------- CHAT ENDPOINT ----------------------
class MessageInput(BaseModel):
    user_input: str

@app.post("/chat")
def get_ai_response(data: MessageInput):
    messages = [
        {"role": "system", "content": "You are a helpful mentor, providing feedback on user's resume's and their overall job search, you can provide helpful information such as what jobs they might be suited for, what they can do to improve their resume, what a cover letter for a job should look like, as well as compare their resume to a job description and requirements. It's useful to note that when a user shares their resume or cover letter with you, you don't keep it too uptight, the user most likely wants a professional but laid-back cover letter or resume, the only things you should look out for are grammatical errors, help them nail their resume and cover letter, but not go too overboard with it. Also bear in mind that some PDF files may have weird spacing, usually this isn't an issue unless the spelling is wrong."},
        {"role": "user", "content": data.user_input}
    ]
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
        return {"response": response.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------- PDF TEXT EXTRACTION ----------------------
@app.post("/extract_pdf_text")
async def extract_pdf_text(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        reader = PyPDF2.PdfReader(io.BytesIO(contents))
        full_text = "".join(page.extract_text() or "" for page in reader.pages)
        return {"text": full_text} if full_text.strip() else {"text": "", "warning": "No extractable text found."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract PDF: {e}")

# ---------------------- RESUME ANALYSIS ----------------------
@app.post("/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        reader = PyPDF2.PdfReader(io.BytesIO(contents))
        text = "".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            return {"response": "No extractable text found in the PDF."}

        messages = [
            {"role": "system", "content": "You are a recruiter AI assistant. Provide feedback on the resume text."},
            {"role": "user", "content": text}
        ]
        response = client.chat.completions.create(model="deepseek-chat", messages=messages)
        return {"response": response.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze PDF: {e}")

# ---------------------- JOB SEARCH (RAPIDAPI WRAPPER) ----------------------
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY2")

@app.get("/api/search")
async def search_jobs(
    query: str = Query(...),
    location: Optional[str] = None,
    country: Optional[str] = None,
    employment_type: Optional[str] = None
):
    if not RAPIDAPI_KEY:
        raise HTTPException(status_code=500, detail="Missing API key")

    enhanced_query = " ".join(filter(None, [query, location, country]))

    params = {
        "query": enhanced_query,
        "page": 1,
        "num_pages": 1,
        "date_posted": "week"
    }
    if location:
        params["job_city"] = location
    if country:
        params["country"] = country
    if employment_type:
        params["employment_types"] = employment_type

    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }

    try:
        res = requests.get("https://jsearch.p.rapidapi.com/search", headers=headers, params=params)
        res.raise_for_status()
        return JSONResponse(content=res.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail="Job search failed.")

# ---------------------- JOB SEARCH (FAST, OPTIMIZED) ----------------------
executor = ThreadPoolExecutor(max_workers=5)

def fetch_jobs_from_jsearch(query: str, page: int = 1):
    headers = {
        "X-RapidAPI-Key": os.getenv("RAPIDAPI_KEY2"),
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    params = {
        "query": query,
        "page": str(page),
        "num_pages": "1",
        "country": "GB",
        "date_posted": "week",
        "work_from_home": "true",
        "fields": ",".join([
            "job_title",
            "employer_name",
            "job_min_salary",
            "job_max_salary",
            "job_employment_type",
            "job_city",
            "job_country",
            "job_description",
            "job_highlights",
            "job_apply_link",
            "employer_logo",
            "job_is_remote",
            "job_posted_at_datetime_utc",
            "job_required_experience"
        ])
    }
    response = requests.get("https://jsearch.p.rapidapi.com/search", headers=headers, params=params)
    response.raise_for_status()
    return response.json().get("data", [])

@app.get("/api/jobs")
async def get_jobs(query: str = Query(...), page: int = 1):
    try:
        jobs = await asyncio.get_event_loop().run_in_executor(executor, fetch_jobs_from_jsearch, query, page)
        results = [{
            "title": job.get("job_title", "N/A"),
            "company": job.get("employer_name", "N/A"),
            "salary": job.get("job_min_salary", "N/A"),
            "max_salary": job.get("job_max_salary", "N/A"),
            "employment_type": job.get("job_employment_type", "N/A"),
            "location": job.get("job_city", "N/A"),
            "country": job.get("job_country", "N/A"),
            "description": job.get("job_description", "N/A"),
            "requirements": job.get("job_required_experience", {}).get("required_qualification", "N/A"),
            "highlights": job.get("job_highlights", {}).get("Qualifications", []),
            "logo": job.get("employer_logo", None),
            "remote": job.get("job_is_remote", False),
            "date_posted": job.get("job_posted_at_datetime_utc", "N/A"),
            "apply_link": job.get("job_apply_link", "N/A")
            } for job in jobs]
        logger.info(f"Fetched {len(results)} jobs for query '{query}' page {page}")
        return {"data": results}
    except requests.HTTPError as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=502, detail="External API error")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
