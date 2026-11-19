# SakhiHealth AI

SakhiHealth AI is a learning/demo full-stack application for women's health education and private wellness journaling. It combines a FastAPI backend, a browser-based frontend, SQLite persistence, and a retrieval system over a curated health knowledge library.

## Features

- User registration and login
- Password hashing and signed bearer-token authentication
- User-scoped wellness journal entries, with deletion
- Mood and symptom summaries
- Women's health knowledge library with source links
- Hybrid document retrieval using TF-IDF and Sentence Transformers semantic similarity
- Synonym expansion and intent-aware topic boosting
- Source references and low-confidence abstention in the assistant workflow
- Basic urgent-symptom routing message
- API health endpoint and interactive Swagger documentation

## Technology

- **Backend:** Python, FastAPI
- **Frontend:** HTML, CSS, JavaScript
- **Storage:** SQLite
- **Retrieval:** scikit-learn TF-IDF, Sentence Transformers (`all-MiniLM-L6-v2`)
- **Testing:** pytest, httpx

## Run on Windows

1. Extract this ZIP.
2. Open the `SakhiHealth-AI` folder in VS Code.
3. Open a terminal in the project root (the folder containing `requirements.txt`).
4. Create and activate a virtual environment:

   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

5. Install dependencies:

   ```cmd
   python -m pip install -r requirements.txt
   ```

6. Set a unique development secret for the current terminal session:

   ```cmd
   set SAKHI_SECRET=replace-this-with-a-long-random-secret
   ```

7. Start the backend:

   ```cmd
   python -m uvicorn backend.app.main:app --reload
   ```

8. Open the application at http://127.0.0.1:8000

API documentation: http://127.0.0.1:8000/docs

You can also run `run_windows.bat` from the project folder. For a custom secret, set `SAKHI_SECRET` in the terminal before launching the batch file.

## Tests

From the project root, run:

```cmd
python -m pytest tests/test_api.py -v
```

Retrieval evaluation scripts:

```cmd
python tests\evaluate_retrieval.py
python tests\test_topic_boost_weights.py
python tests\analyze_rankings.py
```

The retrieval experiments use a small manually labelled evaluation set and should be treated as preliminary results, not proof of general performance or medical accuracy.

## Project structure

```text
SakhiHealth-AI/
├── backend/
│   └── app/
│       └── main.py
├── data/
│   └── health_knowledge.json
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── tests/
│   ├── analyze_rankings.py
│   ├── evaluate_retrieval.py
│   ├── test_api.py
│   └── test_topic_boost_weights.py
├── .env.example
├── .gitignore
├── requirements.txt
├── run_windows.bat
└── README.md
```

## Data, privacy and safety

- The SQLite database is created automatically when the application starts. It is intentionally not included in this GitHub package.
- Do not commit real `.env` files, secrets, personal journal data, or local database files.
- The `.env.example` file is a reference template; the current application reads configuration from environment variables. Set `SAKHI_SECRET` in the terminal or deployment environment.
- This is a student/learning demonstration, not a clinically validated medical product. It does not diagnose conditions, prescribe treatment, or replace professional medical care.
- The urgent-keyword check is basic and is not a reliable emergency detector.
- Retrieval similarity scores are not probabilities, medical confidence scores, or proof that an answer is correct.
- Before any public deployment, review authentication, CORS settings, secret management, privacy, data retention/deletion, HTTPS, rate limiting, and clinical content with appropriate experts.

## Knowledge sources

The knowledge library is stored in `data/health_knowledge.json`. Check the source links and content before relying on or publishing health information.
