# Resume Matcher System

## 🎯 Project Overview

This project is an intelligent system designed for HR users to match candidate resumes with job requirements. It automates the process of scanning a directory of resumes, extracting relevant information, and ranking candidates based on semantic similarity, location compatibility, and project/internship relevance. The system features a backend API built with FastAPI and a user-friendly frontend UI built with Streamlit.

## ✅ Features

*   **Automated Resume Ingestion:** Loads resumes (PDF/DOCX) from a specified directory.
*   **NLP-Powered Resume Parsing:** Extracts candidate name, contact details, skills, locations, projects, and internships using spaCy and regex.
*   **Semantic Matching:** Utilizes Sentence-BERT embeddings to compute semantic similarity between resume content and job descriptions.
*   **Multi-Criteria Ranking:** Ranks candidates based on a weighted combination of:
    *   Semantic similarity score.
    *   Location compatibility.
    *   Skill overlap.
*   **Web Interface for HR:** A Streamlit-based UI for HR users to:
    *   Submit job requirements (title, description, skills, location, salary).
    *   View a ranked list of top-matched candidates.
    *   See highlighted resume sections and detailed match scores.
*   **Backend API:** FastAPI backend with endpoints to:
    *   Accept job postings (`/post-job`).
    *   Return ranked matches and visualization data (`/get-matches/{job_id}`).
*   **Visual Analytics:** Generates and displays plots for:
    *   Top candidate match scores.
    *   Skill overlap heatmaps.
    *   Distribution of semantic similarity scores.
*   **Dummy Data:** Includes sample resumes and job descriptions for testing and demonstration.

## 📦 Technologies Used

*   **Python:** Core programming language.
*   **Backend:**
    *   FastAPI: For building the RESTful API.
    *   Uvicorn: ASGI server for FastAPI.
    *   Pydantic: For data validation in FastAPI.
*   **Frontend:**
    *   Streamlit: For creating the interactive web UI.
    *   Requests: For frontend-backend communication.
*   **NLP & Embeddings:**
    *   spaCy: For resume text parsing and entity extraction.
    *   Sentence-Transformers (Sentence-BERT): For generating semantic embeddings.
    *   Torch: As a dependency for Sentence-Transformers.
*   **Data Handling & Scientific Computing:**
    *   Pandas, NumPy: For data manipulation.
    *   Scikit-learn: For cosine similarity calculation.
    *   Joblib: For caching embeddings.
*   **Text Extraction:**
    *   PyMuPDF (fitz): For extracting text from PDF files.
    *   docx2txt: For extracting text from DOCX files.
*   **Visualization:**
    *   Matplotlib, Seaborn: For generating plots.

## 🗃️ Project Structure

```
resume_matcher/
├── api.py                     # FastAPI backend application
├── data_ingestion.py          # Handles loading and text extraction from resumes
├── embedding_utils.py         # Generates text embeddings using Sentence-BERT
├── matcher.py                 # Core logic for matching and ranking resumes
├── resume_parser.py           # Parses resume text to extract structured information
├── visualization.py           # Generates plots for match results
├── frontend_ui/
│   └── streamlit_app.py       # Streamlit frontend application
├── resumes/                   # Folder for candidate resumes (PDF/DOCX)
│   ├── dummy_resume_data_scientist.pdf
│   ├── dummy_resume_frontend_developer.docx
│   ├── dummy_resume_python_developer.pdf
│   └── generic_resume.pdf
├── generated_plots/           # Directory where visualization plots are saved (created automatically)
├── embeddings_cache/          # Directory where embeddings are cached (created automatically)
├── dummy_jobs.json            # Sample job entries for reference or testing
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## ⚙️ Setup and Installation

1.  **Clone the Repository (if applicable):**
    ```bash
    # git clone <repository_url>
    # cd resume_matcher_project_directory
    ```

2.  **Create a Python Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Navigate to the `resume_matcher` directory (where `requirements.txt` is located) and run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download spaCy NLP Model:**
    After installing dependencies, download the English model for spaCy:
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Prepare Resumes:**
    *   Place candidate resumes (PDF or DOCX files) into the `resume_matcher/resumes/` directory.
    *   Dummy resumes are already provided in this folder for initial testing.

## ▶️ How to Run the Application

The system consists of two main parts: the backend API and the frontend UI. Both need to be running simultaneously.

**1. Run the Backend API (FastAPI):**

*   Navigate to the directory *containing* the `resume_matcher` folder (i.e., the project root if `resume_matcher` is your main app folder).
*   Run the Uvicorn server:
    ```bash
    uvicorn resume_matcher.api:app --reload --port 8000
    ```
    *   `--reload`: Enables auto-reloading for development.
    *   `--port 8000`: Specifies the port (default for the frontend).
*   The API will be accessible at `http://localhost:8000`. You can view the auto-generated API documentation at `http://localhost:8000/docs`.

**2. Run the Frontend UI (Streamlit):**

*   Open a **new terminal window**.
*   Navigate to the directory *containing* the `resume_matcher` folder (project root).
*   Activate the virtual environment if you haven't already in this terminal:
    ```bash
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
*   Run the Streamlit application:
    ```bash
    streamlit run resume_matcher/frontend_ui/streamlit_app.py
    ```
*   The Streamlit UI will typically open in your web browser automatically, usually at `http://localhost:8501`.

## 🚀 How to Use the System

1.  **Ensure both the backend API and frontend UI are running** as described above.
2.  **Open the Streamlit UI** in your web browser (usually `http://localhost:8501`).
3.  In the sidebar of the UI, **fill in the job requirements:**
    *   Job Title
    *   Job Description
    *   Required Skills (comma-separated)
    *   Location (Optional)
    *   Salary Range (Optional)
4.  Click the **"Find Matches"** button.
5.  The system will submit the job to the backend. The UI will show a "processing" message. This might take a moment, especially on the first run as resumes are loaded and embeddings are generated.
6.  Once processing is complete, the UI will display:
    *   A ranked list of top-matched candidates.
    *   Details for each candidate, including match scores and extracted information.
    *   Visualizations like score charts and skill heatmaps.

## 💡 Notes

*   **Caching:** Resume data and embeddings are cached in memory (for API uptime) and on disk (`embeddings_cache/`) to speed up subsequent requests. The first job processing might take longer.
*   **Dummy Data:** The `resumes/` folder contains sample resumes. The `dummy_jobs.json` file contains sample job descriptions that you can use as inspiration for filling out the UI form.
*   **Customization:**
    *   The matching weights in `matcher.py` (`DEFAULT_WEIGHTS`) can be tuned.
    *   The list of predefined skills in `resume_parser.py` (`extract_skills`) can be expanded.
*   **Scalability:** For a production environment with a large number of resumes or frequent use, consider:
    *   A more robust database for storing job submissions and results instead of in-memory storage.
    *   A dedicated task queue (e.g., Celery with Redis/RabbitMQ) for background processing.
    *   Optimizing embedding generation and storage.
    *   Deploying the FastAPI backend and Streamlit frontend using appropriate cloud services or containerization (e.g., Docker).
```
