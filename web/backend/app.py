from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
from app.llm import llm_grade

RULES = [
    # 1. Kiểm tra độ dài tối thiểu (phù hợp với tiểu học)
    {"id": "R01", "desc": "Ít hơn 10 từ", "condition": "word_count_lt", "value": 10,
     "score": -5, "message": "Văn bản quá ngắn"},

    # 2. Có nhắc đến địa danh (phát hiện danh từ riêng viết hoa)
    {"id": "R02", "desc": "Có địa danh", "condition": "has_proper_noun_location",
     "score": 5, "message": "Có nêu địa danh liên quan"},

    # 3. Có nhắc đến nhân vật lịch sử hoặc nhân vật cụ thể
    {"id": "R03", "desc": "Có nhân vật lịch sử", "condition": "has_proper_noun_person",
     "score": 5, "message": "Có nêu nhân vật lịch sử"},

    # 4. Có nêu nguyên nhân, diễn biến hoặc kết quả
    {"id": "R04", "desc": "Có yếu tố nguyên nhân/diễn biến/kết quả", "condition": "has_event_component",
     "score": 5, "message": "Có nêu nguyên nhân, diễn biến hoặc kết quả"},

    # 5. Có yếu tố thời gian (năm, thế kỷ, mốc thời gian)
    {"id": "R05", "desc": "Có yếu tố thời gian", "condition": "has_time_reference",
     "score": 5, "message": "Có mốc thời gian liên quan"}
]



frontend_dir = os.path.join(os.path.dirname(__file__), "../frontend")

# --- Models ---
class InputData(BaseModel):
    question: str
    answer: str
    max_score: int = 3  # mặc định nếu frontend không gửi

class RuleHit(BaseModel):
    id: str
    message: str
    score: int

class Result(BaseModel):
    total_score: int
    rule_hits: List[RuleHit]
    rag_explanation: str

# --- App init ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho mọi domain call API
    allow_methods=["*"],
    allow_headers=["*"],
)

def fake_rag_explanation(text: str, rule_hits: List[RuleHit]):
    # Demo: ghép rule hits thành "giải thích"
    explanations = [f"- {hit.message}" for hit in rule_hits]
    return "Dựa trên phân tích:\n" + "\n".join(explanations) + "\n(Hệ thống RAG sẽ được thêm sau.)"

# --- API ---
@app.post("/score", response_model=Result)
def score_text(data: InputData):
    llm_score, llm_explanation = llm_grade(data.question, data.answer, data.max_score)
    return Result(
        total_score=llm_score,
        rule_hits=[],
        rag_explanation=llm_explanation
    )
# Serve static files (JS, CSS, images)
app.mount("/static", StaticFiles(directory=os.path.join(frontend_dir, "static")), name="static")

# Route cho index.html
@app.get("/")
def read_index():
    return FileResponse(os.path.join(frontend_dir, "static/index.html"))

