import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple, Optional

# Default weights for scoring components - can be tuned
DEFAULT_WEIGHTS = {
    "semantic_similarity": 0.7,
    "location_match": 0.2,
    "skill_overlap": 0.1
}

def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculates cosine similarity between two embeddings."""
    if embedding1 is None or embedding2 is None:
        return 0.0
    if embedding1.ndim == 1:
        embedding1 = embedding1.reshape(1, -1)
    if embedding2.ndim == 1:
        embedding2 = embedding2.reshape(1, -1)

    if not np.any(embedding1) or not np.any(embedding2):
        return 0.0

    try:
        sim_score = cosine_similarity(embedding1, embedding2)[0][0]
        return float(sim_score)
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return 0.0

def check_location_compatibility(resume_locations: List[str], job_location: str) -> bool:
    """
    Checks if there's a location match between resume and job.
    Simple exact match (case-insensitive) for now.
    """
    if not job_location or not resume_locations:
        return False
    normalized_job_loc = job_location.lower().strip()
    for loc in resume_locations:
        if loc.lower().strip() == normalized_job_loc:
            return True
    return False

def calculate_skill_overlap_score(resume_skills: List[str], job_skills: List[str]) -> float:
    """
    Calculates a score based on the overlap of skills between resume and job.
    Normalized by the number of job skills.
    """
    if not job_skills or not resume_skills:
        return 0.0
    norm_resume_skills = set([skill.lower().strip() for skill in resume_skills])
    norm_job_skills = set([skill.lower().strip() for skill in job_skills])
    common_skills = norm_resume_skills.intersection(norm_job_skills)
    if not norm_job_skills:
        return 0.0
    return len(common_skills) / len(norm_job_skills)


def rank_resumes(
    parsed_resumes_data: List[Dict[str, Any]],
    resume_embeddings: Dict[str, np.ndarray],
    job_details: Dict[str, Any],
    job_embedding: np.ndarray,
    weights: Dict[str, float] = None
) -> List[Dict[str, Any]]:
    """
    Ranks resumes based on semantic similarity, location, and skill overlap.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    matched_resumes = []
    job_location_str = job_details.get("location", "")
    job_skills_list = job_details.get("skills", [])
    if isinstance(job_skills_list, str):
        job_skills_list = [s.strip() for s in job_skills_list.split(',') if s.strip()]

    for resume_data in parsed_resumes_data:
        filename = resume_data.get("filename")
        if not filename or filename not in resume_embeddings:
            print(f"Skipping resume {filename or 'Unknown'}: Missing data or embedding.")
            continue

        current_resume_embedding = resume_embeddings[filename]
        semantic_score = calculate_cosine_similarity(current_resume_embedding, job_embedding)
        resume_locations_list = resume_data.get("locations", [])
        location_match_bonus = 1.0 if check_location_compatibility(resume_locations_list, job_location_str) else 0.0
        resume_skills_list = resume_data.get("skills", [])
        skill_score = calculate_skill_overlap_score(resume_skills_list, job_skills_list)

        # Calculate missing skills (skills in job but not in resume)
        norm_job_skills_set = set([s.lower().strip() for s in job_skills_list])
        norm_resume_skills_set = set([s.lower().strip() for s in resume_skills_list])
        job_skills_not_in_resume = list(norm_job_skills_set - norm_resume_skills_set)

        final_score = (
            semantic_score * weights.get("semantic_similarity", 0.7) +
            location_match_bonus * weights.get("location_match", 0.2) +
            skill_score * weights.get("skill_overlap", 0.1)
        )
        final_score = max(0.0, min(final_score, 1.0))

        match_details = {
            "filename": filename,
            "candidate_name": resume_data.get("name", "N/A"),
            "semantic_similarity_score": round(semantic_score, 4),
            "location_match": bool(location_match_bonus),
            "skill_overlap_score": round(skill_score, 4),
            "final_match_score": round(final_score, 4),
            "resume_skills": resume_skills_list,
            "resume_locations": resume_locations_list,
            "job_skills_not_in_resume": job_skills_not_in_resume # Added missing skills
        }
        matched_resumes.append(match_details)

    matched_resumes.sort(key=lambda x: x["final_match_score"], reverse=True)
    return matched_resumes

def rank_jobs_for_resume(resume_details: dict, resume_embedding: np.ndarray, all_jobs_data: dict, weights: Dict[str, float] = None) -> list:
    if weights is None:
        weights = DEFAULT_WEIGHTS

    matched_jobs = []
    if resume_embedding is None or not all_jobs_data:
        return []

    for job_id, job_data in all_jobs_data.items():
        job_details = job_data['details']
        job_embedding = job_data['embedding']

        if job_embedding is None:
            continue

        semantic_score = calculate_cosine_similarity(resume_embedding, job_embedding)
        resume_locations_list = resume_details.get("locations", [])
        job_location_str = job_details.get("location", "")
        location_match = check_location_compatibility(resume_locations_list, job_location_str)
        resume_skills_list = resume_details.get("skills", [])
        job_skills_list = job_details.get("skills", [])
        skill_score = calculate_skill_overlap_score(resume_skills_list, job_skills_list)

        # Calculate skills candidate may need (skills in job but not in resume)
        norm_job_skills_set_fj = set([s.lower().strip() for s in job_skills_list])
        norm_resume_skills_set_fj = set([s.lower().strip() for s in resume_skills_list])
        skills_candidate_may_need = list(norm_job_skills_set_fj - norm_resume_skills_set_fj)

        final_score = (
            semantic_score * weights.get("semantic_similarity", 0.7) +
            (1.0 if location_match else 0.0) * weights.get("location_match", 0.2) +
            skill_score * weights.get("skill_overlap", 0.1)
        )
        final_score = max(0.0, min(final_score, 1.0))

        matched_jobs.append({
            "job_id": job_id,
            "title": job_details.get("title"),
            "description": job_details.get("description"),
            "company": job_details.get("company", "N/A"),
            "location": job_location_str,
            "job_skills": job_skills_list,
            "semantic_similarity_score": round(semantic_score, 4),
            "location_match": location_match,
            "skill_overlap_score": round(skill_score, 4),
            "final_match_score": round(final_score, 4),
            "skills_candidate_may_need": skills_candidate_may_need # Added missing skills
        })

    matched_jobs.sort(key=lambda x: x["final_match_score"], reverse=True)
    return matched_jobs

if __name__ == '__main__':
    print("Running matcher.py directly for testing...")

    sample_parsed_resumes = [
        {
            "filename": "resume1.pdf", "name": "John Doe",
            "locations": ["New York, NY", "San Francisco, CA"],
            "skills": ["python", "machine learning", "django", "fastapi"],
            "cleaned_text": "python developer with machine learning skills and django."
        },
        {
            "filename": "resume4.pdf", "name": "Maria Garcia",
            "locations": ["New York, NY"],
            "skills": ["python", "fastapi", "docker"], # Maria is missing 'machine learning' for the sample job
            "cleaned_text": "Experienced Python and FastAPI developer. Based in New York."
        }
    ]
    EMBED_DIM = 384
    sample_resume_embeddings = {
        "resume1.pdf": np.array([0.1] * EMBED_DIM),
        "resume4.pdf": np.array([0.12] * EMBED_DIM)
    }
    sample_job_details = {
        "id": "job123", "title": "Senior Python Developer",
        "description": "Looking for a python developer with fastapi experience.",
        "location": "New York, NY",
        "skills": ["python", "fastapi", "machine learning", "docker"] # Job requires 'machine learning'
    }
    sample_job_embedding = np.array([0.11] * EMBED_DIM)

    print("\n--- Test: Full Resume Ranking (rank_resumes) with Missing Skills ---")
    ranked_list_resumes = rank_resumes(
        sample_parsed_resumes,
        sample_resume_embeddings,
        sample_job_details,
        sample_job_embedding
    )
    print("Ranked Resumes:")
    for resume in ranked_list_resumes:
        print(
            f"  Filename: {resume['filename']}, Name: {resume['candidate_name']}, "
            f"Final Score: {resume['final_match_score']:.4f}, "
            f"Missing Skills: {resume.get('job_skills_not_in_resume')}"
        )
    # Check resume4 (Maria Garcia) missing skills
    maria_resume_ranked = next((r for r in ranked_list_resumes if r['filename'] == 'resume4.pdf'), None)
    if maria_resume_ranked and 'machine learning' in maria_resume_ranked.get('job_skills_not_in_resume', []):
        print("Missing skills for Maria (resume4.pdf) correctly identified as containing 'machine learning'.")
    else:
        print("Missing skills for Maria (resume4.pdf) NOT correctly identified or resume not found.")


    print("\n--- Test: Full Job Ranking (rank_jobs_for_resume) with Missing Skills ---")
    sample_jobs_data = {
        "job123": {"details": sample_job_details, "embedding": sample_job_embedding},
        "job456": {
            "details": {"id": "job456", "title": "Data Analyst", "description": "Analyze data using SQL and Python.", "location": "Remote", "skills": ["sql", "python", "pandas", "tableau"]},
            "embedding": np.array([0.2] * EMBED_DIM)
            }
    }
    # Test with resume4 (Maria Garcia), who has ["python", "fastapi", "docker"]
    # Job456 requires "sql", "pandas", "tableau" which Maria doesn't have.
    ranked_list_jobs = rank_jobs_for_resume(sample_parsed_resumes[1], sample_resume_embeddings["resume4.pdf"], sample_jobs_data)
    print(f"Ranked Jobs for Resume: {sample_parsed_resumes[1]['filename']} ({sample_parsed_resumes[1]['name']})")
    for job in ranked_list_jobs:
        print(
            f"  Job ID: {job['job_id']}, Title: {job['title']}, "
            f"Final Score: {job['final_match_score']:.4f}, "
            f"Skills Candidate May Need: {job.get('skills_candidate_may_need')}"
        )
    job456_ranked = next((j for j in ranked_list_jobs if j['job_id'] == 'job456'), None)
    if job456_ranked:
        expected_missing_for_job456 = {'sql', 'pandas', 'tableau'}
        actual_missing_for_job456 = set(job456_ranked.get('skills_candidate_may_need', []))
        if expected_missing_for_job456.issubset(actual_missing_for_job456): # Check if expected are present
             print("Skills candidate may need for job456 correctly identified for Maria.")
        else:
             print(f"Skills candidate may need for job456 for Maria are {actual_missing_for_job456}, expected to include {expected_missing_for_job456}.")
    else:
        print("Job456 not found in ranked jobs for Maria.")

    print("\nMatcher test finished.")
