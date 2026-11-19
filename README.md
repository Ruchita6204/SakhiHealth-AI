# SakhiHealth AI

SakhiHealth AI is a learning and demonstration full-stack application designed to support women's health education and private wellness journaling. It combines a FastAPI backend, a browser-based frontend, SQLite database storage, and a hybrid information retrieval system using a curated women's health knowledge library.

## Features

- User registration and login
- Password hashing and signed bearer-token authentication
- User-specific wellness journal entries with deletion functionality
- Mood and symptom summaries
- Women's health knowledge library with source references
- Hybrid document retrieval using TF-IDF and Sentence Transformer semantic similarity
- Synonym expansion and intent-aware topic boosting
- Source references and low-confidence abstention in the assistant workflow
- Basic urgent-symptom routing message
- API health endpoint and interactive Swagger documentation

## Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Database | SQLite |
| Keyword Retrieval | TF-IDF, scikit-learn |
| Semantic Retrieval | Sentence Transformers |
| Semantic Model | all-MiniLM-L6-v2 |
| Testing | pytest, httpx |

## Installation and Setup (Windows)

### Step 1: Extract the project

Extract the ZIP file and open the `SakhiHealth-AI` folder in Visual Studio Code.

### Step 2: Open the terminal

Open the VS Code terminal in the project root directory (the folder containing `requirements.txt`).

### Step 3: Create a virtual environment

```cmd
python -m venv .venv
```

### Step 4: Activate the virtual environment

```cmd
.venv\Scripts\activate
```

### Step 5: Install dependencies

```cmd
python -m pip install -r requirements.txt
```

### Step 6: Configure the application secret

Set a unique secret for your development session:

```cmd
set SAKHI_SECRET=replace-this-with-a-long-random-secret
```

Use a secure random value instead of the example above. Do not publish your actual secret.

### Step 7: Start the backend server

```cmd
python -m uvicorn backend.app.main:app --reload
```

### Step 8: Open the application

- Application: http://127.0.0.1:8000
- API documentation: http://127.0.0.1:8000/docs
- API health check: http://127.0.0.1:8000/api/health

You can also use `run_windows.bat` from the project folder to start the application. Set `SAKHI_SECRET` in the terminal before running the batch file.

## Testing

Run the API tests from the project root:

```cmd
python -m pytest tests/test_api.py -v
```

### Retrieval Evaluation

The project includes the following retrieval evaluation scripts:

```cmd
python tests\evaluate_retrieval.py
```

```cmd
python tests\test_topic_boost_weights.py
```

```cmd
python tests\analyze_rankings.py
```

These scripts evaluate document retrieval performance, topic-boost configurations, and per-query ranking behaviour.

The retrieval experiments use a small manually labelled evaluation dataset. Their results are preliminary and should not be interpreted as proof of general performance or medical accuracy.

## Project Structure

```text
SakhiHealth-AI/
│
├── backend/
│   ├── __init__.py
│   └── app/
│       ├── __init__.py
│       └── main.py
│
├── data/
│   └── health_knowledge.json
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── tests/
│   ├── analyze_rankings.py
│   ├── evaluate_retrieval.py
│   ├── test_api.py
│   └── test_topic_boost_weights.py
│
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── run_windows.bat
```

## Data, Privacy and Safety

- The SQLite database is created automatically when the application starts. It is intentionally excluded from this GitHub package.
- Do not commit actual `.env` files, passwords, API keys, authentication secrets, personal journal data, or local database files.
- The `.env.example` file provides a reference configuration. The application reads configuration from environment variables. Set `SAKHI_SECRET` in your terminal or deployment environment.
- SakhiHealth AI is a student learning and demonstration project, not a clinically validated medical product.
- The application does not diagnose medical conditions, prescribe treatments, or replace professional medical advice.
- The urgent-keyword check is basic and must not be treated as a reliable emergency detection or medical triage system.
- Retrieval similarity scores are not probabilities, medical confidence scores, or proof of answer correctness.
- Before public deployment, review authentication, CORS settings, secret management, privacy, data retention and deletion, HTTPS, rate limiting, and clinical content with appropriate experts.

## Knowledge Sources

The women's health knowledge library is maintained in:

`data/health_knowledge.json`

It contains curated educational information and source references. Review the original sources and verify the content before relying on or publishing health information.

## Project Purpose

SakhiHealth AI demonstrates how information retrieval, web technologies, and structured health knowledge can be combined to build an accessible women's health education and wellness journaling platform.
