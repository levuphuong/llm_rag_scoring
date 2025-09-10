import re
from llm import grade_exam_with_rag
import time


def parse_exam_txt(file_path: str):
    """
    Parse an exam text file into a list of questions with answer options.
    Each question block starts with 'Câu <number>.' or 'Câu <number> (points).'
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split by question pattern: "Câu <number>." or "Câu <number> (x points)."
    parts = re.split(r"\n*Câu\s+\d+(?:\s*\([^)]*\))?\s*\.", text)

    questions = []
    # The first element is usually empty because regex split starts from beginning
    for idx, part in enumerate(parts[1:], 1):
        part = part.strip()
        if not part:
            continue

        # Collect all lines, e.g., question text + options (A., B., C., D.)
        lines = [l.strip() for l in part.split("\n") if l.strip()]
        q_text = lines[0]       # the question
        options = lines[1:]     # the answer choices

        # Standardize format
        question_text = f"Câu {idx}. {q_text}\n" + "\n".join(options)

        questions.append({
            "question": question_text,
            "student_answer": ""  # will be filled later
        })

    return questions


if __name__ == "__main__":
    # ----- Read exam file -----
    file_path = "./dataset/test1.txt"   # path to exam .txt file
    persist_dir = "./db/sgk"               # ChromaDB persistence directory (for RAG)
    course = "lichsu_dialy_4"

    exam = parse_exam_txt(file_path)

    # Example: student answers (hardcoded here for testing)
    student_answers = ["C", "A", "D"]  # adjust depending on your test case
    for i, ans in enumerate(student_answers):
        exam[i]["student_answer"] = ans

    # # ====== Run without RAG ======
    # print("\n=== Results without RAG ===")
    # start = time.time()
    # total, results = grade_exam_with_rag(exam, persist_dir, use_rag=False)
    # print(f"Total score: {total}/{len(exam)}")
    # for r in results:
    #     print(f"Question {r['question_id']}: {r['feedback']}")
    # end = time.time()
    # print(f"\nExecution time: {end - start:.2f} seconds")

    # ====== Run with RAG ======
    print("=== Results with RAG ===")
    start = time.time()
    total, results = grade_exam_with_rag(exam, persist_dir, course, use_rag=True)
    print(f"Total score: {total}/{len(exam)}")
    for r in results:
        print(f"Question {r['question_id']}: {r['feedback']}")
    end = time.time()
    print(f"\nExecution time: {end - start:.2f} seconds")
