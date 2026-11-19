# SakhiHealth AI – Women's Health Education & Wellness Platform

SakhiHealth AI is a learning and demonstration full-stack application designed to support women's health education and private wellness journaling. The platform combines a curated women's health knowledge library, hybrid machine learning-based information retrieval, and personalized wellness tracking to provide accessible educational information.

The application integrates a **FastAPI backend, HTML/CSS/JavaScript frontend, SQLite database, and a hybrid information retrieval system** using TF-IDF and Sentence Transformers.

SakhiHealth AI aims to demonstrate how artificial intelligence, information retrieval, and web technologies can be combined to make women's health education more accessible while encouraging awareness, self-reflection, and informed health-related learning.

---

## 1. Key Features

### 1.1 User Authentication
- User registration and login.
- Password hashing using PBKDF2-HMAC-SHA256 with randomly generated salts.
- Custom HMAC-signed bearer-token authentication.
- Token expiration and user authentication checks.
- User-specific access to wellness journal records.

### 1.2 AI-Powered Health Information Assistant
- Interactive AI-based question-answering interface.
- Hybrid information retrieval using TF-IDF and Sentence Transformers.
- Semantic similarity-based document matching.
- Keyword-based retrieval and synonym expansion.
- Intent-aware topic detection and ranking.
- Relevant source references for retrieved information.
- Low-confidence response abstention to reduce unsupported answers.
- TF-IDF fallback when the semantic model is unavailable.
- Basic urgent-symptom keyword screening.

### 1.3 Women's Health Learning Library
- Curated women's health educational resources.
- Information organized by health-related topics.
- Source titles and original reference links.
- Educational content covering menstrual health, PCOS, symptoms, mental wellbeing, and other women's health topics.

### 1.4 Wellness Journal
- User-specific wellness journal entries.
- Daily mood and symptom tracking.
- Personal notes and journal entry management.
- Journal entry deletion functionality.
- SQLite-based data storage.

### 1.5 Wellness Analytics
- Mood distribution summaries.
- Symptom frequency analysis.
- Total journal entry count.
- Recent journal activity summaries.

### 1.6 Feedback System
- User feedback submission.
- Helpful and unhelpful response tracking.
- Feedback storage for application evaluation.

---

## 2. Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| Frontend | HTML5, CSS3, JavaScript |
| Database | SQLite |
| API Framework | FastAPI |
| Keyword Retrieval | TF-IDF |
| Machine Learning | Scikit-learn |
| Semantic Retrieval | Sentence Transformers |
| Pre-trained Model | all-MiniLM-L6-v2 |
| Deep Learning Framework | PyTorch |
| Data Validation | Pydantic |
| API Server | Uvicorn |
| Testing | Pytest, HTTPX |
| Version Control | Git and GitHub |

---

## 3. System Architecture

SakhiHealth AI follows a modular client-server architecture consisting of the frontend, backend, database, and AI-based information retrieval components.

### 3.1 Frontend Layer
The frontend provides the user interface and application navigation.

Technologies:
- HTML
- CSS
- JavaScript

Responsibilities:
- User registration and login interfaces.
- AI assistant interaction.
- Learning library presentation.
- Wellness journal management.
- Mood and symptom analytics.
- Communication with backend API endpoints.

### 3.2 Backend Layer
The backend is developed using Python and FastAPI.

Responsibilities:
- User authentication and authorization.
- Request validation.
- API endpoint management.
- Knowledge library loading.
- AI-based document retrieval.
- Journal and feedback management.
- Database operations.

### 3.3 Database Layer
SQLite is used for local application data storage.

The database maintains three primary tables:

1. **Users:** Stores user account information and password hashes.
2. **Journals:** Stores user-specific wellness journal entries.
3. **Feedback:** Stores feedback submitted by users.

The database tables and required indexes are created automatically when the application initializes.

### 3.4 AI and Information Retrieval Layer
The AI retrieval component combines keyword-based and semantic document retrieval.

The system uses:
- TF-IDF vectorization.
- Sentence Transformer embeddings.
- Cosine similarity.
- Synonym expansion.
- Intent detection.
- Topic-based ranking adjustments.

The retrieved documents are used to construct educational responses with supporting source references.

---

## 4. Hybrid Information Retrieval Methodology

SakhiHealth AI implements a hybrid information retrieval approach to identify relevant educational documents from its curated knowledge library.

### 4.1 TF-IDF Retrieval

Term Frequency-Inverse Document Frequency (TF-IDF) represents documents and user queries as numerical vectors.

The system uses:
- English stop-word removal.
- Unigrams and bigrams.
- Lowercase text processing.
- Sublinear term-frequency scaling.
- Cosine similarity for document matching.

### 4.2 Semantic Retrieval

The semantic retrieval component uses the pre-trained Sentence Transformer model:

`sentence-transformers/all-MiniLM-L6-v2`

The model converts knowledge documents and user queries into numerical embeddings.

Normalized embeddings are compared using cosine similarity through dot-product computation.

This allows the system to identify semantically related documents even when the exact keywords are different.

The model may require an internet connection for its initial download. Once cached, it can be loaded from the local model cache.

### 4.3 Hybrid Score Calculation

The system combines TF-IDF and semantic similarity scores using the following weighted approach:

**Hybrid Score = (0.35 × TF-IDF Score) + (0.65 × Semantic Score)**

The semantic component receives a higher weight to emphasize contextual similarity.

If the semantic model is unavailable, the system uses TF-IDF retrieval as a fallback.

### 4.4 Synonym Expansion

Synonym expansion improves retrieval for commonly used alternative health-related terms.

Examples include:

| User Term | Expanded Terms |
|---|---|
| PCOD | PCOS, Polycystic Ovary Syndrome |
| Periods | Menstrual, Menstruation, Cycle |
| Cramps | Menstrual Pain, Abdominal Pain |
| Mood Swings | Mood Changes, Emotional Wellbeing |
| Menopause | Menopausal, Life-Course Health |

### 4.5 Intent-Aware Topic Boosting

The system identifies user-query intents using predefined keywords.

Supported topic categories include:
- Mental wellbeing.
- Menstrual pain.
- Menstrual health.
- PCOS.
- Endometriosis.
- Menopause.

Relevant topic matches receive an additional ranking adjustment of 0.10.

### 4.6 Confidence Threshold and Abstention

The retrieval system applies minimum relevance thresholds before returning documents.

- Semantic retrieval threshold: 0.30.
- TF-IDF fallback threshold: 0.04.

The system returns relevant documents only when the retrieval score satisfies the configured threshold.

If sufficiently relevant information is unavailable, the assistant abstains rather than generating an unsupported response.

**Important:** Retrieval similarity scores are ranking signals, not probabilities of correctness or medical confidence scores.

---

## 5. API Endpoints

The application exposes REST API endpoints through FastAPI.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Backend and AI system health check |
| POST | `/api/auth/register` | Register a new user |
| POST | `/api/auth/login` | Authenticate an existing user |
| GET | `/api/me` | Retrieve authenticated user profile |
| GET | `/api/knowledge` | Retrieve the health knowledge library |
| POST | `/api/ask` | Submit a question to Sakhi AI |
| POST | `/api/feedback` | Submit assistant feedback |
| GET | `/api/journal` | Retrieve user journal entries |
| POST | `/api/journal` | Create a journal entry |
| DELETE | `/api/journal/{entry_id}` | Delete a journal entry |
| GET | `/api/analytics` | Retrieve wellness analytics |

Protected endpoints require a valid bearer token.

### API Documentation

FastAPI provides interactive API documentation through Swagger UI.

- Application: http://127.0.0.1:8000
- Swagger documentation: http://127.0.0.1:8000/docs
- Health endpoint: http://127.0.0.1:8000/api/health

---

## 6. Installation and Setup (Windows)

Follow these instructions to run SakhiHealth AI locally.

### Step 1: Extract the Project

Download or clone the repository and extract the project folder.

Open the `SakhiHealth-AI` folder in Visual Studio Code.

### Step 2: Open the Terminal

Open the VS Code terminal in the project root directory containing `requirements.txt`.

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

```cmd
.venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 6: Configure the Application Secret

Set a unique development secret in the Windows CMD terminal:

```cmd
set SAKHI_SECRET=your-long-random-secret
```

Replace the example with a securely generated random value.

The secret is required for signing and validating authentication tokens.

Do not publish your actual secret or upload your `.env` file to GitHub.

### Step 7: Start the Backend Server

```bash
python -m uvicorn backend.app.main:app --reload
```

Wait until the terminal displays:

```text
Application startup complete.
```

During initialization, the application loads the knowledge library and prepares the TF-IDF and semantic retrieval models.

### Step 8: Open the Application

Open the following addresses in your browser:

**Frontend:**
http://127.0.0.1:8000

**API Documentation:**
http://127.0.0.1:8000/docs

**API Health Check:**
http://127.0.0.1:8000/api/health

Alternatively, use the provided `run_windows.bat` file to start the application after setting the required environment variable.

---

## 7. Testing and Evaluation

The project includes API testing and retrieval evaluation scripts.

### 7.1 API Testing

Run the API test suite from the project root:

```bash
python -m pytest tests/test_api.py -v
```

This test suite can be used to verify the application's API functionality.

### 7.2 Retrieval Evaluation

The project includes three evaluation scripts:

**Retrieval Performance Evaluation**

```bash
python tests\evaluate_retrieval.py
```

**Topic Boost Weight Evaluation**

```bash
python tests\test_topic_boost_weights.py
```

**Document Ranking Analysis**

```bash
python tests\analyze_rankings.py
```

These scripts evaluate:
- Relevant document retrieval.
- Ranking behaviour.
- Topic-boost configurations.
- Retrieval performance across manually labelled queries.

### 7.3 Evaluation Limitations

The retrieval experiments use a small manually labelled evaluation dataset.

The reported results are preliminary and intended for development and comparative experimentation.

They should not be interpreted as evidence of general clinical accuracy, medical reliability, or performance across all women's health questions.

---

## 8. Project Structure

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

The SQLite database is generated locally during application initialization and is not included in the GitHub project package.

---

## 9. Data Privacy, Security and Safety

SakhiHealth AI is a student learning and demonstration project. Privacy and responsible handling of health-related information are important considerations.

### Data Privacy
- User journal entries are associated with individual user accounts.
- SQLite is used for local application data storage.
- Actual local database files are excluded from the GitHub package.
- Environment files and authentication secrets must not be committed.
- Personal journal information should not be shared publicly.

### Security Considerations
- Passwords are hashed using PBKDF2-HMAC-SHA256 with random salts.
- Authentication tokens are signed using HMAC-SHA256.
- Protected API endpoints require authentication.
- User-specific journal operations are restricted to the authenticated account.

Before public deployment, additional security measures should be reviewed, including token handling, CORS restrictions, HTTPS, rate limiting, secure secret management, database protection, and privacy controls.

### Medical Safety
- The platform provides general educational information only.
- It does not diagnose medical conditions.
- It does not prescribe medication or treatment.
- It does not replace qualified healthcare professionals.
- The urgent-symptom screening is based on basic keyword matching and is not a clinically validated emergency detection or triage system.
- Knowledge content and original references should be reviewed and verified before publication or use.

SakhiHealth AI must not be used as a substitute for professional medical assessment.

---

## 10. Knowledge Sources

The curated women's health knowledge library is maintained in:

```text
data/health_knowledge.json
```

The file contains structured educational documents with fields such as:
- Document ID.
- Topic.
- Title.
- Educational content.
- Source information.
- Original reference URL.
- Relevant keywords.

The knowledge library is used by the hybrid retrieval engine to identify relevant educational information.

Original sources should be reviewed and verified periodically to maintain content accuracy and relevance.

---

## 11. Current Development Verification

During local development, the following components were successfully initialized and verified:

| Component | Development Status |
|---|---|
| Python virtual environment | Verified |
| Required dependencies | Installed |
| FastAPI backend | Started successfully |
| SQLite database initialization | Initialized |
| Knowledge documents | 13 loaded |
| TF-IDF model | Ready |
| Sentence Transformer model | Ready |
| Hybrid retrieval engine | Ready |
| API health endpoint | Returned `status: ok` |
| Frontend application | Opened successfully in Microsoft Edge |

Further functional testing of authentication, AI question answering, journal operations, analytics, and feedback should be completed using the provided test procedures.

---

## 12. Future Scope

Potential future improvements include:

- Expanding the curated women's health knowledge library.
- Improving retrieval accuracy through larger and more diverse evaluation datasets.
- Adding multilingual health education support.
- Improving semantic search and query understanding.
- Introducing stronger privacy and security controls.
- Adding more comprehensive wellness analytics.
- Improving accessibility and mobile responsiveness.
- Conducting expert reviews of educational health content.
- Performing larger-scale usability and retrieval evaluations.

These are proposed improvements and are not claimed as existing features.

---

## 13. Project Purpose

SakhiHealth AI demonstrates the practical application of artificial intelligence, machine learning, information retrieval, and full-stack web development in the domain of women's health education.

By integrating a curated knowledge library, hybrid retrieval mechanisms, and private wellness journaling, the project explores how technology can support accessible health education and personal wellness awareness.

The project is intended for academic demonstration, software development learning, and further experimentation with responsible AI-based information retrieval systems.

---

**Project:** SakhiHealth AI  
**Category:** AI/ML and Full-Stack Web Development  
**Backend:** Python and FastAPI  
**Frontend:** HTML, CSS and JavaScript  
**Database:** SQLite  
**AI Approach:** Hybrid Information Retrieval  
**Model:** Sentence Transformers (all-MiniLM-L6-v2)

