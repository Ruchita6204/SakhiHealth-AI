
from __future__ import annotations
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
from pathlib import Path


import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

# This file is expected at: backend/app/main.py
# parents[2] resolves to the project root: SakhiHealth-AI-FullStack
ROOT = Path(__file__).resolve().parents[2]

DB_PATH = Path(
    os.getenv("SAKHI_DB", str(ROOT / "sakhihealth.db"))
).resolve()

KNOWLEDGE_PATH = ROOT / "data" / "health_knowledge.json"
FRONTEND = ROOT / "frontend"

SECRET = os.getenv(
    "SAKHI_SECRET",
    "development-only-change-this-secret"
)

TOKEN_LIFETIME_SECONDS = 7 * 24 * 60 * 60

print("\n========== SAKHIHEALTH AI STARTUP ==========")
print("Project root:", ROOT)
print("Database:", DB_PATH)
print("Knowledge file:", KNOWLEDGE_PATH)

if SECRET == "development-only-change-this-secret":
    print("WARNING: Set a unique SAKHI_SECRET in your .env before deployment.")


# ============================================================
# 2. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="SakhiHealth AI API",
    version="2.1.0",
    description=(
        "Women's health education and private wellness journaling. "
        "This application does not diagnose conditions or prescribe treatment."
    ),
)

# Local development configuration.
# Before deployment, replace "*" with your actual frontend origin(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 3. DATABASE
# ============================================================

def db() -> sqlite3.Connection:
    """Create a SQLite connection with row access and foreign keys."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db() -> None:
    """Create required database tables and indexes if they do not exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with db() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS journals (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                entry_date TEXT NOT NULL,
                mood TEXT NOT NULL,
                symptoms TEXT NOT NULL,
                notes TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id)
                    REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                question TEXT NOT NULL,
                helpful INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id)
                    REFERENCES users(id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_journals_user_date
            ON journals(user_id, entry_date);
            """
        )


init_db()


# ============================================================
# 4. LOAD AND VALIDATE KNOWLEDGE LIBRARY
# ============================================================

def load_knowledge() -> list[dict]:
    if not KNOWLEDGE_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge file not found: {KNOWLEDGE_PATH}"
        )

    try:
        documents = json.loads(
            KNOWLEDGE_PATH.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON in knowledge file: {exc}"
        ) from exc

    if not isinstance(documents, list):
        raise ValueError("Knowledge file must contain a JSON list.")

    required_fields = ["id", "topic", "title", "content"]
    seen_ids = set()

    for index, item in enumerate(documents):
        if not isinstance(item, dict):
            raise ValueError(
                f"Document at index {index} is not a JSON object."
            )

        missing = [
            field for field in required_fields
            if not item.get(field)
        ]

        if missing:
            raise ValueError(
                f"Document at index {index} is missing fields: {missing}"
            )

        document_id = str(item["id"])

        if document_id in seen_ids:
            raise ValueError(f"Duplicate knowledge document ID: {document_id}")

        seen_ids.add(document_id)
        item["id"] = document_id

    print("Knowledge documents loaded:", len(documents))
    print("Document IDs:", [item["id"] for item in documents])

    return documents


KNOWLEDGE = load_knowledge()


# ============================================================
# 5. PREPARE KNOWLEDGE TEXT
# ============================================================

def prepare_text(item: dict) -> str:
    keywords = item.get("keywords", "")

    if isinstance(keywords, list):
        keywords = " ".join(str(value) for value in keywords)

    return " ".join(
        [
            str(item.get("topic", "")),
            str(item.get("title", "")),
            str(item.get("content", "")),
            str(keywords),
        ]
    ).strip()


CORPUS = [prepare_text(item) for item in KNOWLEDGE]


# ============================================================
# 6. TF-IDF MODEL
# ============================================================

VECTORIZER = None
TFIDF_MATRIX = None

if CORPUS:
    VECTORIZER = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        lowercase=True,
        sublinear_tf=True,
    )

    TFIDF_MATRIX = VECTORIZER.fit_transform(CORPUS)

    print("TF-IDF model trained successfully.")
    print("TF-IDF indexed documents:", TFIDF_MATRIX.shape[0])


# ============================================================
# 7. SYNONYM EXPANSION
# ============================================================

SYNONYMS = {
    "pcod": "pcos polycystic ovary syndrome",
    "polycystic ovary disease": "pcos polycystic ovary syndrome",
    "period": "menstrual menstruation cycle",
    "periods": "menstrual menstruation cycle",
    "cramps": "menstrual pain abdominal pain",
    "irregular periods": "irregular menstrual cycle infrequent periods",
    "mood swings": "mood changes emotional wellbeing",
    "period pain": "menstrual cramps pain",
    "menopause": "menopausal life course health",
    "feeling low": "sadness emotional wellbeing mental health",
    "feeling anxious": "anxiety stress mental wellbeing",
}


def expand_question(question: str) -> str:
    expanded = question.lower()

    for term, replacement in SYNONYMS.items():
        if term in expanded:
            expanded += " " + replacement

    return expanded


# ============================================================
# 8. SENTENCE TRANSFORMER SEMANTIC MODEL
# ============================================================

SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SEMANTIC_MODEL = None
DOCUMENT_EMBEDDINGS = None
SEMANTIC_LOAD_ERROR = None

if CORPUS:
    try:
        print("Loading semantic model:", SEMANTIC_MODEL_NAME)

        SEMANTIC_MODEL = SentenceTransformer(
            SEMANTIC_MODEL_NAME
        )

        DOCUMENT_EMBEDDINGS = SEMANTIC_MODEL.encode(
            CORPUS,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        print("Semantic model loaded successfully.")
        print("Semantic documents indexed:", len(DOCUMENT_EMBEDDINGS))

    except Exception as exc:
        SEMANTIC_LOAD_ERROR = str(exc)
        SEMANTIC_MODEL = None
        DOCUMENT_EMBEDDINGS = None

        print("Semantic model unavailable:", exc)
        print("Using TF-IDF fallback.")


# ============================================================
# 9. QUESTION INTENT AND HYBRID DOCUMENT RETRIEVAL
# ============================================================

INTENT_KEYWORDS = {
    "mental_wellbeing": [
        "stress", "anxiety", "anxious", "mood", "mood swings",
        "sadness", "emotional", "mental health", "mental wellbeing",
        "irritability", "feeling sad", "feeling low", "overthinking",
        "depression", "emotional distress",
    ],
    "menstrual_pain": [
        "cramps", "period pain", "menstrual pain",
        "abdominal pain", "pain during periods",
    ],
    "menstrual_health": [
        "menstrual cycle", "periods", "menstruation",
        "irregular periods", "heavy bleeding",
    ],
    "pcos": [
        "pcos", "pcod", "polycystic ovary",
        "ovarian syndrome",
    ],
    "endometriosis": [
        "endometriosis",
    ],
    "menopause": [
        "menopause", "menopausal", "hot flashes",
    ],
}

TOPIC_MAPPING = {
    "mental_wellbeing": [
        "mental wellbeing", "mental health",
    ],
    "menstrual_pain": [
        "pain and symptoms", "menstrual pain",
    ],
    "menstrual_health": [
        "menstrual health basics", "when to seek care",
    ],
    "pcos": [
        "pcos awareness",
    ],
    "endometriosis": [
        "endometriosis awareness",
    ],
    "menopause": [
        "menopause and life-course health",
    ],
}


def detect_intents(question: str) -> list[str]:
    question_lower = question.lower()
    detected = []

    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in question_lower for keyword in keywords):
            detected.append(intent)

    return detected


def retrieve_documents(
    question: str,
    top_k: int = 3,
) -> tuple[list[dict], list[float]]:
    """Retrieve relevant documents using hybrid ML and topic intent boosts."""

    if not KNOWLEDGE or not question.strip():
        return [], []

    expanded_question = expand_question(question)

    # 1. TF-IDF scores
    tfidf_scores = np.zeros(len(KNOWLEDGE), dtype=float)

    if VECTORIZER is not None and TFIDF_MATRIX is not None:
        query_vector = VECTORIZER.transform([expanded_question])

        tfidf_scores = cosine_similarity(
            query_vector,
            TFIDF_MATRIX,
        ).ravel()

    # 2. Semantic similarity scores
    semantic_scores = np.zeros(len(KNOWLEDGE), dtype=float)

    if SEMANTIC_MODEL is not None and DOCUMENT_EMBEDDINGS is not None:
        query_embedding = SEMANTIC_MODEL.encode(
            [expanded_question],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        # Embeddings are normalized, so dot product is cosine similarity.
        semantic_scores = np.dot(
            DOCUMENT_EMBEDDINGS,
            query_embedding,
        )

        # Keep semantic similarity in its natural cosine range.
        semantic_scores = np.clip(semantic_scores, 0.0, 1.0)

    # 3. Combine scores
    if SEMANTIC_MODEL is not None:
        hybrid_scores = (
            0.35 * tfidf_scores
            + 0.65 * semantic_scores
        )
    else:
        hybrid_scores = tfidf_scores.copy()

    if not len(hybrid_scores):
        return [], []

    # 4. Intent-aware topic boost
    detected_intents = detect_intents(question)

    for index, item in enumerate(KNOWLEDGE):
        topic = str(item.get("topic", "")).lower()

        for intent in detected_intents:
            matching_topics = TOPIC_MAPPING.get(intent, [])

            if any(target in topic for target in matching_topics):
                hybrid_scores[index] += 0.10

    # 5. Rank documents
    order = np.argsort(hybrid_scores)[::-1][:max(1, top_k)]

    if len(order) == 0:
        return [], []

    best_score = float(hybrid_scores[order[0]])

    threshold = 0.30 if SEMANTIC_MODEL is not None else 0.04

    if best_score < threshold:
        return [], []

    # 6. Keep documents close to the best match
    selected = []
    selected_scores = []

    minimum_score = max(threshold, best_score * 0.70)

    for index in order:
        score = float(hybrid_scores[index])

        if score >= minimum_score:
            selected.append(KNOWLEDGE[int(index)])
            selected_scores.append(score)

    return selected, selected_scores
# ============================================================
# 10. REQUEST MODELS
# ============================================================

class AuthIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=8, max_length=128)


class LoginIn(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class JournalIn(BaseModel):
    entry_date: str
    mood: str = Field(min_length=1, max_length=30)
    symptoms: list[str] = Field(default_factory=list)
    notes: str = Field(default="", max_length=4000)


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class FeedbackIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    helpful: bool


# ============================================================
# 11. PASSWORD HASHING
# ============================================================

def pw_hash(password: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(16)

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        180000,
    ).hex()

    return f"{salt}${digest}"


def verify(password: str, stored: str) -> bool:
    try:
        salt, expected_digest = stored.split("$", 1)

        actual_digest = pw_hash(
            password,
            salt,
        ).split("$", 1)[1]

        return hmac.compare_digest(
            actual_digest,
            expected_digest,
        )

    except (ValueError, AttributeError):
        return False


# ============================================================
# 12. AUTHENTICATION TOKEN
# ============================================================

def make_token(uid: int) -> str:
    payload_data = {
        "uid": uid,
        "exp": int(time.time()) + TOKEN_LIFETIME_SECONDS,
    }

    payload = base64.urlsafe_b64encode(
        json.dumps(payload_data).encode("utf-8")
    ).decode("utf-8").rstrip("=")

    signature = hmac.new(
        SECRET.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return payload + "." + signature

bearer_scheme = HTTPBearer(auto_error=False)
def current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
):
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Please sign in."
        )

    token = credentials.credentials

    try:
        payload, sig = token.split(".", 1)

        expected = hmac.new(
            SECRET.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(sig, expected):
            raise ValueError("Invalid token")

        data = json.loads(
            base64.urlsafe_b64decode(
                payload + "=" * ((4 - len(payload) % 4) % 4)
            )
        )

        if data["exp"] < time.time():
            raise ValueError("Expired token")

        with db() as c:
            user = c.execute(
                "SELECT id, email, name FROM users WHERE id=?",
                (data["uid"],)
            ).fetchone()

        if not user:
            raise ValueError("User not found")

        return dict(user)

    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Session expired. Please sign in again."
        )


# ============================================================
# 13. HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():
    if SEMANTIC_MODEL is not None:
        ml_status = "Hybrid TF-IDF + Sentence Transformers"
    elif VECTORIZER is not None:
        ml_status = "TF-IDF retrieval"
    else:
        ml_status = "unavailable"

    return {
        "status": "ok",
        "service": "SakhiHealth AI",
        "knowledge_documents": len(KNOWLEDGE),
        "knowledge_file_exists": KNOWLEDGE_PATH.exists(),
        "ml": ml_status,
        "tfidf_ready": TFIDF_MATRIX is not None,
        "semantic_model_ready": SEMANTIC_MODEL is not None,
        "hybrid_ready": (
            TFIDF_MATRIX is not None
            and SEMANTIC_MODEL is not None
        ),
    }


# ============================================================
# 14. USER REGISTRATION
# ============================================================

@app.post("/api/auth/register")
def register(data: AuthIn):
    email = data.email.strip().lower()
    name = data.name.strip()

    email_pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

    if not re.match(email_pattern, email):
        raise HTTPException(
            status_code=400,
            detail="Enter a valid email address.",
        )

    if not name:
        raise HTTPException(
            status_code=400,
            detail="Name cannot be empty.",
        )

    try:
        with db() as con:
            cursor = con.execute(
                """
                INSERT INTO users (email, name, password_hash)
                VALUES (?, ?, ?)
                """,
                (
                    email,
                    name,
                    pw_hash(data.password),
                ),
            )

            user_id = cursor.lastrowid

    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists.",
        )

    return {
        "token": make_token(user_id),
        "user": {
            "id": user_id,
            "email": email,
            "name": name,
        },
    }


# ============================================================
# 15. USER LOGIN
# ============================================================

@app.post("/api/auth/login")
def login(data: LoginIn):
    email = data.email.strip().lower()

    with db() as con:
        user = con.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,),
        ).fetchone()

    if user is None or not verify(data.password, user["password_hash"]):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password.",
        )

    return {
        "token": make_token(user["id"]),
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
        },
    }


# ============================================================
# 16. CURRENT USER PROFILE
# ============================================================

@app.get("/api/me")
def me(user=Depends(current_user)):
    return user


# ============================================================
# 17. KNOWLEDGE LIBRARY API
# ============================================================

@app.get("/api/knowledge")
def knowledge():
    return [
        {
            "id": item.get("id"),
            "topic": item.get("topic"),
            "title": item.get("title"),
            "source": item.get("source"),
            "url": item.get("url"),
            "content": item.get("content"),
        }
        for item in KNOWLEDGE
    ]


# ============================================================
# 18. ASK SAKHI AI
# ============================================================

@app.post("/api/ask")
def ask(
    data: AskIn,
    user=Depends(current_user),
):
    question = data.question.strip()

    # Basic emergency phrase screening.
    # This is not a complete emergency detection system.
    urgent_pattern = (
        r"\b(chest pain|difficulty breathing|can't breathe|"
        r"cannot breathe|heavy bleeding|fainting|suicid|"
        r"overdose|severe allergic reaction)\b"
    )

    if re.search(urgent_pattern, question, re.IGNORECASE):
        return {
            "answer": (
                "Your message may describe an urgent situation. "
                "Please seek immediate medical help from local emergency "
                "services or the nearest hospital. If you are in India, "
                "call 112 for emergency assistance. This assistant cannot "
                "assess emergencies."
            ),
            "sources": [],
            "confidence": 0,
            "disclaimer": "Educational support only; not a diagnosis.",
        }

    if not KNOWLEDGE:
        return {
            "answer": "The learning library is currently unavailable.",
            "sources": [],
            "confidence": 0,
            "disclaimer": "Please consult a qualified healthcare professional.",
        }

    selected, scores = retrieve_documents(question)

    if not selected:
        return {
            "answer": (
                "I could not find sufficiently relevant information in "
                "the current learning library. I do not want to guess. "
                "Please rephrase your question or consult a qualified "
                "healthcare professional for personal medical advice."
            ),
            "sources": [],
            "confidence": 0,
            "disclaimer": "For education only, not diagnosis or treatment.",
        }

    # Build answer from retrieved knowledge only.
    snippets = []

    for item in selected[:2]:
        content = str(item.get("content", "")).strip()

        if content:
            snippets.append(content)

    if not snippets:
        return {
            "answer": (
                "I could not find enough readable information to answer "
                "that question. Please consult a qualified healthcare professional."
            ),
            "sources": [],
            "confidence": 0,
            "disclaimer": "For education only, not diagnosis or treatment.",
        }

    answer = " ".join(snippets)

    answer += (
        "\n\nThis is general educational information. "
        "It cannot diagnose a condition or replace advice "
        "from a qualified healthcare professional."
    )

    sources = [
        {
            "id": item.get("id"),
            "title": item.get("title"),
            "source": item.get("source"),
            "url": item.get("url"),
        }
        for item in selected
    ]

    return {
        "answer": answer,
        "sources": sources,
        "confidence": round(float(max(scores)), 3) if scores else 0,
        "disclaimer": "For education only, not diagnosis or treatment.",
        "retrieval_method": (
            "hybrid"
            if SEMANTIC_MODEL is not None
            else "tfidf_fallback"
        ),
    }
# ============================================================
# 19. ASSISTANT FEEDBACK
# ============================================================

@app.post("/api/feedback")
def save_feedback(
    data: FeedbackIn,
    user=Depends(current_user),
):
    with db() as con:
        cursor = con.execute(
            """
            INSERT INTO feedback (user_id, question, helpful)
            VALUES (?, ?, ?)
            """,
            (
                user["id"],
                data.question.strip(),
                1 if data.helpful else 0,
            ),
        )

    return {
        "saved": True,
        "feedback_id": cursor.lastrowid,
    }


# ============================================================
# 20. JOURNAL - GET ENTRIES
# ============================================================

@app.get("/api/journal")
def list_journal(user=Depends(current_user)):
    with db() as con:
        rows = con.execute(
            """
            SELECT *
            FROM journals
            WHERE user_id = ?
            ORDER BY entry_date DESC, id DESC
            """,
            (user["id"],),
        ).fetchall()

    result = []

    for row in rows:
        item = dict(row)

        try:
            item["symptoms"] = json.loads(item["symptoms"])
        except (TypeError, json.JSONDecodeError):
            item["symptoms"] = []

        result.append(item)

    return result


# ============================================================
# 21. JOURNAL - ADD ENTRY
# ============================================================

@app.post("/api/journal")
def add_journal(
    data: JournalIn,
    user=Depends(current_user),
):
    try:
        time.strptime(data.entry_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Date must use YYYY-MM-DD format.",
        )

    clean_symptoms = [
        symptom.strip()
        for symptom in data.symptoms
        if isinstance(symptom, str) and symptom.strip()
    ]

    with db() as con:
        cursor = con.execute(
            """
            INSERT INTO journals (
                user_id, entry_date, mood, symptoms, notes
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user["id"],
                data.entry_date,
                data.mood.strip(),
                json.dumps(clean_symptoms),
                data.notes.strip(),
            ),
        )

        row = con.execute(
            "SELECT * FROM journals WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()

    result = dict(row)
    result["symptoms"] = json.loads(result["symptoms"])

    return result


# ============================================================
# 22. JOURNAL - DELETE ENTRY
# ============================================================

@app.delete("/api/journal/{entry_id}")
def delete_journal(
    entry_id: int,
    user=Depends(current_user),
):
    with db() as con:
        cursor = con.execute(
            """
            DELETE FROM journals
            WHERE id = ? AND user_id = ?
            """,
            (entry_id, user["id"]),
        )

    if cursor.rowcount == 0:
        raise HTTPException(
            status_code=404,
            detail="Journal entry not found.",
        )

    return {"deleted": True}


# ============================================================
# 23. JOURNAL ANALYTICS
# ============================================================

@app.get("/api/analytics")
def analytics(user=Depends(current_user)):
    with db() as con:
        rows = con.execute(
            """
            SELECT mood, symptoms, entry_date
            FROM journals
            WHERE user_id = ?
            ORDER BY entry_date DESC, id DESC
            """,
            (user["id"],),
        ).fetchall()

    moods = {}
    symptoms = {}

    for row in rows:
        mood = row["mood"]
        moods[mood] = moods.get(mood, 0) + 1

        try:
            entry_symptoms = json.loads(row["symptoms"])
        except (TypeError, json.JSONDecodeError):
            entry_symptoms = []

        for symptom in entry_symptoms:
            symptoms[symptom] = symptoms.get(symptom, 0) + 1

    return {
        "total_entries": len(rows),
        "moods": moods,
        "symptoms": symptoms,
        "recent_entries": [
            row["entry_date"] for row in rows[:7]
        ],
    }


# ============================================================
# 24. FRONTEND
# ============================================================

if not FRONTEND.exists():
    raise FileNotFoundError(
        f"Frontend folder not found: {FRONTEND}"
    )

INDEX_FILE = FRONTEND / "index.html"

if not INDEX_FILE.exists():
    raise FileNotFoundError(
        f"Frontend index.html not found: {INDEX_FILE}"
    )

app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND)),
    name="static",
)


@app.get("/")
def home():
    return FileResponse(str(INDEX_FILE))


# ============================================================
# 25. STARTUP SUMMARY
# ============================================================

print("\nSakhiHealth AI backend initialized.")
print("Knowledge documents:", len(KNOWLEDGE))
print("TF-IDF ready:", TFIDF_MATRIX is not None)
print("Semantic model ready:", SEMANTIC_MODEL is not None)
print(
    "Hybrid ML ready:",
    TFIDF_MATRIX is not None and SEMANTIC_MODEL is not None,
)
print("============================================\n")