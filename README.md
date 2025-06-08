# Web-Based Job Matching Platform

This project is a web-based job matching platform built with Python, Flask, and various NLP/ML libraries. It allows candidates to upload resumes and HR professionals to post jobs (via CSV) and find suitable matches.

## Features

*   **Candidate Portal**:
    *   Upload resumes (PDF, DOCX, TXT).
    *   View job postings matched to the uploaded resume based on semantic similarity, skill overlap, and location.
    *   See a list of skills from job descriptions that are not present in their resume.
*   **HR Portal**:
    *   Upload job descriptions in bulk via a CSV file.
    *   View candidate resumes matched to each job posting.
    *   See a list of job skills not present in a candidate's resume.
*   **Backend**:
    *   Uses Sentence Transformers (`all-MiniLM-L6-v2`) for generating text embeddings for resumes and job descriptions.
    *   Matching logic combines cosine similarity of embeddings with heuristics for skill overlap and location compatibility.
    *   Resumes are stored in `resume_matcher/resumes/`.
    *   Job data (from CSV) and embeddings are stored in memory for the current session.
    *   Embeddings for resumes and jobs are cached in `resume_matcher/embeddings_cache/` to speed up subsequent processing.
*   **Frontend**:
    *   Built with Flask and Jinja2 templates.

## Prerequisites

*   Python (3.8+ recommended)
*   pip (Python package installer)
*   Git (for cloning, if applicable)

## Setup Instructions

1.  **Clone the Repository (if applicable)**:
    \`\`\`bash
    git clone <repository-url>
    cd <repository-directory>
    \`\`\`

2.  **Create and Activate a Virtual Environment (Recommended)**:
    \`\`\`bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    \`\`\`

3.  **Install Dependencies**:
    Navigate to the `resume_matcher` directory (if your `requirements.txt` is there) or ensure your `requirements.txt` is at the project root. The current setup has `resume_matcher/requirements.txt`.
    \`\`\`bash
    pip install -r resume_matcher/requirements.txt
    \`\`\`

4.  **Download spaCy NLP Model**:
    The resume parser uses a spaCy model. Download it by running:
    \`\`\`bash
    python -m spacy download en_core_web_sm
    \`\`\`

## Directory Structure


\`\`\`
.
├── app.py                      # Main Flask application
├── job_posts.csv               # Sample CSV for HR job uploads (place in root)
├── labeled_matches.csv         # Dummy labeled data for ML model training
├── README.md                   # This file
├── resume_matcher/             # Core application package
│   ├── __init__.py
│   ├── data_ingestion.py       # Resume file text extraction
│   ├── embedding_utils.py      # Text embedding generation
│   ├── matcher.py              # Matching algorithms
│   ├── resume_parser.py        # Resume text parsing (skills, name, etc.)
│   ├── train_model.py          # Placeholder ML model training script
│   ├── models/                 # Stores trained ML models (e.g., svm_match_classifier.joblib)
│   ├── embeddings_cache/       # Cached embeddings for performance
│   ├── resumes/                # Stores uploaded/dummy resumes
│   └── requirements.txt        # Python dependencies
├── static/                     # Static files (CSS, JS - currently minimal)
├── templates/                  # Jinja2 HTML templates
│   ├── index.html              # Homepage
│   ├── hr_index.html           # HR dashboard
│   ├── hr_upload.html          # HR CSV upload form
│   ├── hr_matches_overview.html # HR list of jobs
│   ├── hr_specific_job_matches.html # HR view of candidates for a job
│   ├── candidate_index.html    # Candidate dashboard
│   ├── candidate_upload.html   # Candidate resume upload form
│   └── candidate_matches.html  # Candidate view of jobs for their resume
└── uploads/                    # Temporary storage for uploaded files (e.g., CSVs)
\`\`\`

## Running the Application

1.  Ensure you are in the project root directory (where `app.py` is located).
2.  Make sure your virtual environment is activated.
3.  Run the Flask development server:

    \`\`\`bash
    python app.py
    \`\`\`

4.  Open your web browser and navigate to `http://localhost:8000` (or `http://0.0.0.0:8000`).

## How to Use

### For HR Professionals:

1.  Navigate to the "For HR Professionals" link from the homepage, or go directly to `/hr`.
2.  Click on "Upload New Jobs CSV".
3.  Prepare a CSV file with columns: `job_id`, `title`, `description`, `skills` (comma-separated). Optional columns: `location`, `salary`.
    *   A sample `job_posts.csv` is provided in the project root.
4.  Upload the CSV file. The system will process the jobs and generate embeddings.
5.  From the HR Dashboard, click "View/Match Jobs".
6.  Click "View Matches" for a specific job to see a list of ranked candidate resumes.
    *   The system will load resumes from the `resume_matcher/resumes/` directory. Ensure some resumes are present there. Sample resumes are included.
    *   Matches are ranked by a composite score including semantic similarity, skill overlap, and location preference.
    *   You can also see which job skills are not present in a candidate's resume.

### For Candidates:

1.  Navigate to the "For Candidates" link from the homepage, or go directly to `/candidate`.
2.  Click on "Upload Your Resume".
3.  Upload your resume (PDF, DOCX, or TXT format).
4.  After successful upload and processing, you will be redirected to a page showing job postings that match your resume.
    *   Jobs are ranked based on relevance to your resume.
    *   You can see which skills required by a job might be missing from your resume.

## Machine Learning Model (SVM/ANN)

*   A placeholder script `resume_matcher/train_model.py` is included for training an SVM (or ANN) classifier.
*   This script is **not fully integrated** into the matching logic of the live application. The current matching uses cosine similarity and heuristics.
*   To train a custom model:
    1.  Prepare a labeled dataset (`labeled_matches.csv` in the project root) with resume-job pairs and match labels (1 for good match, 0 for not a good match). You would also need to generate features for these pairs, typically derived from their embeddings and other factors. The current `train_model.py` creates dummy features if the CSV doesn't contain them.
    2.  Run the training script: `python resume_matcher/train_model.py`.
    3.  The trained model will be saved in `resume_matcher/models/`.
    4.  To use this model, the `rank_resumes` and `rank_jobs_for_resume` functions in `resume_matcher/matcher.py` would need to be modified to load the model and use its predictions/scores.

## Future Enhancements / Notes

*   The current application stores job data and processed resume data in memory, meaning it resets when the Flask app restarts. For persistence, a database would be needed.
*   Error handling can be further improved.
*   The UI is basic; CSS and JavaScript could be added for a richer experience.
*   Resume parsing for skills can be made more sophisticated (e.g., using NLP techniques beyond keyword matching).
