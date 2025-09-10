from openai import OpenAI
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb
import re, json




client = OpenAI(api_key="YOUR OPEN AI TOKEN")

# ----- Tiền xử lý -----
def preprocess(text: str):
    text = text.lower().strip()
    sentences = sent_tokenize(text)
    return sentences

# ----- RAG mô phỏng -----
def rag_retrieval(query: str, documents: List[str]):
    best_doc = ""
    max_overlap = 0
    query_words = set(query.split())
    for doc in documents:
        doc_words = set(doc.split())
        overlap = len(query_words.intersection(doc_words))
        if overlap > max_overlap:
            max_overlap = overlap
            best_doc = doc
    return best_doc

# ----- Gọi OpenAI LLM -----
def llm_grade(question: str, answer: str, max_score = 3):

    """
    Chấm điểm dựa trên base rule tổng quát (hình thức + nội dung).
    """
    base_rules = [
        "Có nêu nhân vật hoặc lực lượng lãnh đạo chính.",
        "Có nêu đối thủ hoặc phe bị đánh bại.",
        "Có nêu kết quả hoặc ý nghĩa chung của sự kiện.",
        "Có ít nhất một thông tin mô tả sự kiện (thời gian, địa điểm hoặc chiến thuật).",
        "Trình bày dễ hiểu."
    ]

    rules_text = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(base_rules, 1)])

    prompt = f"""
    Bạn là hệ thống chấm điểm dựa trên các tiêu chí tổng quát sau:
    {rules_text}

    Cho điểm từ 0 đến {max_score}, mỗi tiêu chí đạt được thì cộng 1 điểm.
    Trả về kết quả dạng JSON:
    {{
    "score": <số nguyên>,
    "explanation": "<giải thích>"
    }}

    Câu hỏi:
    \"\"\"{question}\"\"\"

    Câu trả lời của học sinh:
    \"\"\"{answer}\"\"\" 
    """



    response = client.chat.completions.create(
        model="gpt-4o-mini",
        # model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content

    # parse JSON trả về
    import json
    try:
        result = json.loads(content)
        score = result.get("score", 0)
        explanation = result.get("explanation", "")
    except json.JSONDecodeError:
        score = 0
        explanation = content

    return score, explanation

def query_chroma(query, db_path="./chroma_db", collection_name="sgk_collection", top_k=5):
    # Load DB
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(collection_name)

    # Embedding query
    model = SentenceTransformer("intfloat/multilingual-e5-base")
    q_embed = model.encode([f"query: {query}"]).tolist()

    # Search
    results = collection.query(query_embeddings=q_embed, n_results=top_k)

    if results and "documents" in results and results["documents"][0]:
        return results["documents"][0][0]
    return ""


def safe_parse_json(content: str):
    """Thử lấy JSON trong output GPT (kể cả khi có ```json``` hoặc text thừa)."""
    # Thử parse thẳng
    try:
        return json.loads(content)
    except Exception:
        pass

    # Thử tìm đoạn trong code fence ```json ... ```
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # Thử cắt từ dấu { đến }
    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end > start:
        snippet = content[start:end+1]
        try:
            return json.loads(snippet)
        except Exception:
            pass

    return None

def grade_mcq_with_rag(question, student_answer, persist_dir, collection, max_score=1, use_rag=True):
    context = ""
    if use_rag and persist_dir:
        context = query_chroma(question, persist_dir, collection)
        print (f"Context: {context}")
    else:
        print ("Khong Co RAG")

    if context:
        context_text = f"Ngữ cảnh:\n{context}\n\n"
    else:
        context_text = ""

    prompt = f"""
    Bạn là giáo viên lịch sử. {context_text}
    Câu hỏi:
    {question}

    Trả về JSON:
    {{
      "correct_answer": "A/B/C/D",
      "explanation": "<giải thích ngắn gọn>"
    }}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content

    parsed = safe_parse_json(content)
    if parsed:
        correct_answer = parsed.get("correct_answer", "").strip().upper()
        explanation = parsed.get("explanation", "")
    else:
        correct_answer = "?"
        explanation = content  # fallback debug

    student_answer = (student_answer or "").strip().upper()

    if student_answer == correct_answer:
        score = max_score
        feedback = f"✅ Đúng! Bạn chọn {student_answer}. {explanation}"
    else:
        score = 0
        feedback = f"❌ Sai. Bạn chọn {student_answer}, đáp án đúng là {correct_answer}. {explanation}"

    print ("get answer")
    return score, correct_answer, feedback

# ----- Chấm nhiều câu -----
def grade_exam_with_rag(questions: List[Dict], persist_dir: str, collection: str, max_score_per_question=1, use_rag=True):
    total_score = 0
    results = []

    for i, q in enumerate(questions, 1):
        score, correct_answer, feedback = grade_mcq_with_rag(
            q["question"], q["student_answer"], persist_dir, collection,  max_score=max_score_per_question, use_rag=use_rag
        )
        total_score += score
        results.append({
            "question_id": i,
            "question": q["question"],
            "student_answer": q["student_answer"],
            "correct_answer": correct_answer,
            "score": score,
            "feedback": feedback
        })

    return total_score, results