from domain.schemas import InterviewState


def create_initial_state() -> InterviewState:
    return {
        "history": [],
        "weakness_prefix": "",
        "is_ready": False,
        "user_input": "",
        "extracted_text": "",
        "has_file": False,
        "decision": {},
        "response": "",
        "short_summary": "[]",
        "turn_count": 0,
        "is_generating_report": False,
        "should_end": False,
        "file_name": "",
        "jd_text": "",
        "resume_text": ""
    }
