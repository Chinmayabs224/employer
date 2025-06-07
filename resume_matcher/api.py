# resume_matcher/api.py

import os
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import importlib # For dynamic imports if needed, or direct

# Dynamically determine project root or assume a structure
# This helps locate sibling modules like data_ingestion, resume_parser etc.
# Assuming api.py is in resume_matcher/ and other .py files are siblings.
try:
    from .data_ingestion import load_resumes_from_directory
    from .resume_parser import parse_resume_text
    from .embedding_utils import generate_resume_embeddings, generate_job_description_embedding
    from .matcher import rank_resumes, calculate_cosine_similarity as calculate_cosine_similarity_from_matcher
    from .visualization import plot_top_candidate_scores, plot_skills_heatmap_table, plot_semantic_similarity_distribution
except ImportError:
    # Fallback for cases where the script might be run in a way that . imports don't work as expected
    # This can happen if running `python api.py` directly from within resume_matcher
    # For uvicorn from project root `uvicorn resume_matcher.api:app` the . imports should work.
    print("Relative imports failed. Attempting to adjust path or use absolute imports if modules are installed.")
    # This is a common issue depending on how the app is run.
    # For now, we rely on the execution context (like uvicorn from parent dir) to handle module resolution.
    # If running `python api.py` for testing, ensure PYTHONPATH includes the parent of `resume_matcher`.
    # Or, ensure all modules are installed in the environment if they were structured as a package.
    # For this project structure, running with `uvicorn resume_matcher.api:app` from the directory
    # containing `resume_matcher/` is the typical way.

    # For robust local script execution if needed (though uvicorn is preferred):
    # import sys
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.dirname(script_dir) # Assuming api.py is in resume_matcher/
    # if project_root not in sys.path:
    #    sys.path.insert(0, project_root)
    # from resume_matcher.data_ingestion import load_resumes_from_directory
    # from resume_matcher.resume_parser import parse_resume_text
    # ... and so on for other modules
    raise # Re-raise for now, as this should be configured by how uvicorn runs it.


# Initialize FastAPI app
app = FastAPI(
    title="Resume Matcher API",
    description="API for matching resumes to job descriptions.",
    version="0.1.0"
)

# In-memory storage for job details and results
jobs_db: Dict[str, Dict[str, Any]] = {}
resumes_cache = {
    "parsed_data": [],
    "embeddings": {}
}
resumes_loaded_and_processed = False # Flag to track if initial resume load has happened

# --- Pydantic Models ---
class JobInput(BaseModel):
    title: str = Field(..., example="Software Engineer")
    description: str = Field(..., example="Develop and maintain web applications...")
    skills: List[str] = Field(default_factory=list, example=["python", "fastapi", "react"])
    location: Optional[str] = Field(None, example="New York, NY")
    salary: Optional[str] = Field(None, example="100k - 120k USD")

class JobResponse(BaseModel):
    job_id: str
    message: str
    status_url: Optional[str] = None

class MatchResult(BaseModel):
    filename: str
    candidate_name: Optional[str] = None
    final_match_score: float
    semantic_similarity_score: float
    location_match: bool
    skill_overlap_score: float
    resume_skills: Optional[List[str]] = None
    resume_locations: Optional[List[str]] = None

class JobResultResponse(BaseModel):
    job_id: str
    job_details: JobInput
    status: str
    ranked_matches: Optional[List[MatchResult]] = None
    visualizations: Optional[Dict[str, str]] = None
    error_message: Optional[str] = None

# --- Helper Functions ---
def process_job_request_task(job_id: str, job_input_data: JobInput):
    global resumes_cache, resumes_loaded_and_processed

    try:
        print(f"Background task started for job ID: {job_id}")
        jobs_db[job_id]["status"] = "PROCESSING"

        # Correct path to resumes directory: should be relative to the project root.
        # data_ingestion.py expects 'resumes/' to be a child of its own directory.
        # If api.py and data_ingestion.py are in 'resume_matcher/', and 'resumes' is also in 'resume_matcher/',
        # then the path "resumes/" is correct for data_ingestion.
        resumes_dir_relative_to_module = "resumes/"

        if not resumes_loaded_and_processed or not resumes_cache["parsed_data"]:
            print("Loading and processing resumes for the first time or cache is empty...")
            raw_resumes = load_resumes_from_directory(directory_path=resumes_dir_relative_to_module)

            parsed_resumes_list = []
            for r_data in raw_resumes:
                if r_data.get("text"):
                    parsed_info = parse_resume_text(r_data["text"])
                    parsed_info["filename"] = r_data["filename"]
                    parsed_resumes_list.append(parsed_info)
                else:
                    print(f"Warning: No text extracted for {r_data.get('filename', 'Unknown file')}. Skipping.")

            if not parsed_resumes_list:
                msg = "No resumes found or parsed in the specified directory. Ensure 'resumes/' directory exists and contains valid PDF/DOCX files."
                print(f"Error for job {job_id}: {msg}")
                jobs_db[job_id]["status"] = "FAILED"
                jobs_db[job_id]["error_message"] = msg
                return

            resumes_cache["parsed_data"] = parsed_resumes_list
            resumes_cache["embeddings"] = generate_resume_embeddings(parsed_resumes_list, force_regenerate=False)
            resumes_loaded_and_processed = True
            print(f"Resumes loaded: {len(parsed_resumes_list)} parsed, embeddings generated.")
        else:
            print("Using cached resume data and embeddings.")

        job_embedding = generate_job_description_embedding(job_input_data.model_dump(), force_regenerate=True)
        if job_embedding is None:
            jobs_db[job_id]["status"] = "FAILED"
            jobs_db[job_id]["error_message"] = "Could not generate job embedding (check job description and skills)."
            return

        ranked_candidates = rank_resumes(
            parsed_resumes_data=resumes_cache["parsed_data"],
            resume_embeddings=resumes_cache["embeddings"],
            job_details=job_input_data.model_dump(),
            job_embedding=job_embedding
        )
        jobs_db[job_id]["ranked_matches"] = [MatchResult(**match) for match in ranked_candidates]

        vis_paths = {}
        if ranked_candidates:
            # For semantic similarity distribution, calculate against all resumes
            all_semantic_scores = []
            if resumes_cache["embeddings"] and job_embedding is not None:
                 for res_filename, res_embed in resumes_cache["embeddings"].items():
                    # Use the imported calculate_cosine_similarity_from_matcher
                    score = calculate_cosine_similarity_from_matcher(res_embed, job_embedding)
                    all_semantic_scores.append(score)

            plot1 = plot_top_candidate_scores(ranked_candidates, top_n=10, job_id=job_id)
            if plot1: vis_paths["top_candidate_scores_plot"] = os.path.basename(plot1)

            plot2 = plot_skills_heatmap_table(ranked_candidates, job_input_data.skills, top_n=5, job_id=job_id)
            if plot2: vis_paths["skills_heatmap_plot"] = os.path.basename(plot2)

            if all_semantic_scores: # Only plot if scores were calculated
                plot3 = plot_semantic_similarity_distribution(all_semantic_scores, job_id=job_id)
                if plot3: vis_paths["similarity_distribution_plot"] = os.path.basename(plot3)
            else: # Fallback to scores from ranked candidates if full calc fails
                temp_scores = [m.semantic_similarity_score for m in jobs_db[job_id]["ranked_matches"]]
                if temp_scores:
                    plot3 = plot_semantic_similarity_distribution(temp_scores, job_id=job_id)
                    if plot3: vis_paths["similarity_distribution_plot"] = os.path.basename(plot3)


        jobs_db[job_id]["visualizations"] = vis_paths
        jobs_db[job_id]["status"] = "COMPLETED"
        print(f"Successfully processed job ID: {job_id}")

    except Exception as e:
        import traceback
        print(f"Unhandled error processing job {job_id}: {e}\n{traceback.format_exc()}")
        jobs_db[job_id]["status"] = "FAILED"
        jobs_db[job_id]["error_message"] = f"An unexpected error occurred: {str(e)}"

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    # Ensure base directories required by modules exist relative to api.py's location
    # visualization.py creates 'generated_plots'
    # embedding_utils.py creates 'embeddings_cache'
    # data_ingestion.py uses 'resumes'

    # Assuming api.py is in resume_matcher/, these dirs will be resume_matcher/generated_plots, etc.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir, "resumes"), exist_ok=True)
    os.makedirs(os.path.join(script_dir, "generated_plots"), exist_ok=True) # For visualization.py
    os.makedirs(os.path.join(script_dir, "embeddings_cache"), exist_ok=True) # For embedding_utils.py

    resumes_path = os.path.join(script_dir, "resumes")

    # Create a dummy resume for testing if resumes folder is empty
    if not os.listdir(resumes_path):
        dummy_pdf_path = os.path.join(resumes_path, "api_dummy_resume.pdf")
        try:
            import fitz # PyMuPDF
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 72), "API Dummy Resume for Test. Skills: Python, Java. Lives in Test City, USA.")
            doc.save(dummy_pdf_path)
            doc.close()
            print(f"Created dummy resume for API testing: {dummy_pdf_path}")
        except Exception as e:
            print(f"Could not create dummy PDF for API testing (PyMuPDF error or other): {e}")

    print("Resume Matcher API started. Necessary directories checked/created.")
    print(f"Resumes will be loaded from: {os.path.abspath(resumes_path)}")


@app.post("/post-job", response_model=JobResponse, status_code=202)
async def post_job(job_input: JobInput, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs_db[job_id] = {
        "job_details": job_input.model_dump(),
        "status": "PENDING",
        "ranked_matches": None,
        "visualizations": None,
        "error_message": None
    }

    background_tasks.add_task(process_job_request_task, job_id, job_input)

    return JobResponse(
        job_id=job_id,
        message="Job received and is being processed in the background.",
        status_url=f"/get-matches/{job_id}" # Provide the full path for client
    )

@app.get("/get-matches/{job_id}", response_model=JobResultResponse)
async def get_matches(job_id: str):
    job_data = jobs_db.get(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found.")

    return JobResultResponse(
        job_id=job_id,
        job_details=JobInput(**job_data["job_details"]),
        status=job_data["status"],
        ranked_matches=job_data.get("ranked_matches", []),
        visualizations=job_data.get("visualizations", {}),
        error_message=job_data.get("error_message")
    )

# To serve plot files directly from FastAPI (optional, Streamlit can also access filesystem)
# Ensure the PLOTS_DIR used by visualization.py is accessible here.
# visualization.py saves plots in 'generated_plots' relative to its own location.
# If api.py and visualization.py are in the same dir (resume_matcher/), this path is correct.
plots_static_dir_name = "generated_plots"
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_abs_path = os.path.join(script_dir, plots_static_dir_name)

# Check if StaticFiles is already imported, if not, do it here
try:
    from fastapi.staticfiles import StaticFiles
    # Mount only if the directory actually exists (it's created by visualization.py or startup)
    if os.path.exists(plots_abs_path):
        app.mount(f"/{plots_static_dir_name}", StaticFiles(directory=plots_abs_path), name="plots")
        print(f"Serving static files from /{plots_static_dir_name} (maps to: {plots_abs_path})")
    else:
        print(f"Directory {plots_abs_path} not found. Static files for plots will not be served by FastAPI directly unless created.")
except ImportError:
    print("fastapi.staticfiles not found, cannot serve plots directly via FastAPI.")


if __name__ == "__main__":
    print("To run this API, use Uvicorn from the parent directory of 'resume_matcher':")
    print("Example: uvicorn resume_matcher.api:app --reload --port 8000")
    # For development, you might need to add the project's parent directory to PYTHONPATH
    # if you encounter module not found errors for data_ingestion, etc. when running uvicorn.
    # This usually means Python can't find the 'resume_matcher' package.
    # Running uvicorn from one level above 'resume_matcher' (e.g. from 'project_root/')
    # with `uvicorn resume_matcher.api:app` should make 'resume_matcher' discoverable as a package.
