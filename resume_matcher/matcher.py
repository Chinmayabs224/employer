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

    # Ensure embeddings are not all zeros, which can lead to NaN or errors.
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
    job_location is a string, resume_locations is a list of strings.
    """
    if not job_location or not resume_locations:
        return False
    # Normalize for comparison
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

    # Normalize skills for comparison (lowercase, strip)
    norm_resume_skills = set([skill.lower().strip() for skill in resume_skills])
    norm_job_skills = set([skill.lower().strip() for skill in job_skills])

    common_skills = norm_resume_skills.intersection(norm_job_skills)

    if not norm_job_skills: # Avoid division by zero if job_skills is empty after normalization
        return 0.0

    return len(common_skills) / len(norm_job_skills)


def rank_resumes(
    parsed_resumes_data: List[Dict[str, Any]], # From resume_parser
    resume_embeddings: Dict[str, np.ndarray], # From embedding_utils {filename: embedding}
    job_details: Dict[str, Any], # Contains job title, desc, skills, location
    job_embedding: np.ndarray,
    weights: Dict[str, float] = None
) -> List[Dict[str, Any]]:
    """
    Ranks resumes based on semantic similarity, location, and skill overlap.

    Args:
        parsed_resumes_data: List of dictionaries, each containing parsed info for a resume.
                             Expected keys: 'filename', 'locations', 'skills'.
        resume_embeddings: Dictionary mapping resume filenames to their embeddings.
        job_details: Dictionary with job info, including 'location' (str) and 'skills' (List[str]).
        job_embedding: Embedding vector for the job description.
        weights: Dictionary to weigh different scoring components.

    Returns:
        A list of dictionaries, each representing a matched resume with its scores,
        sorted by the final match score in descending order.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    matched_resumes = []

    job_location_str = job_details.get("location", "")
    job_skills_list = job_details.get("skills", [])
    if isinstance(job_skills_list, str): # If skills is a comma separated string
        job_skills_list = [s.strip() for s in job_skills_list.split(',') if s.strip()]


    for resume_data in parsed_resumes_data:
        filename = resume_data.get("filename")
        if not filename or filename not in resume_embeddings:
            print(f"Skipping resume {filename or 'Unknown'}: Missing data or embedding.")
            continue

        current_resume_embedding = resume_embeddings[filename]

        # 1. Semantic Similarity Score
        semantic_score = calculate_cosine_similarity(current_resume_embedding, job_embedding)

        # 2. Location Match Score (binary for now, could be a score if using proximity)
        resume_locations_list = resume_data.get("locations", [])
        location_match_bonus = 1.0 if check_location_compatibility(resume_locations_list, job_location_str) else 0.0

        # 3. Skill Overlap Score
        resume_skills_list = resume_data.get("skills", [])
        skill_score = calculate_skill_overlap_score(resume_skills_list, job_skills_list)

        # Combine scores using weights
        final_score = (
            semantic_score * weights.get("semantic_similarity", 0.7) +
            location_match_bonus * weights.get("location_match", 0.2) + # location_match is 0 or 1
            skill_score * weights.get("skill_overlap", 0.1)
        )

        # Normalize final score to be between 0 and 1 (assuming weights sum to 1)
        # If weights don't sum to 1, this normalization might need adjustment.
        # For now, it's a weighted sum. Max possible score is sum of weights.
        # If we want it strictly 0-1, divide by sum of weights.
        # sum_of_weights = sum(weights.values())
        # if sum_of_weights > 0:
        #     final_score = final_score / sum_of_weights
        # else: # Avoid division by zero if all weights are zero
        #     final_score = 0.0
        # Clamping the score between 0 and 1, as cosine similarity is -1 to 1, but we treat it as 0 to 1.
        # And other scores are 0 to 1.
        final_score = max(0.0, min(final_score, 1.0))


        match_details = {
            "filename": filename,
            "candidate_name": resume_data.get("name", "N/A"),
            "semantic_similarity_score": round(semantic_score, 4),
            "location_match": bool(location_match_bonus),
            "skill_overlap_score": round(skill_score, 4),
            "final_match_score": round(final_score, 4),
            "resume_skills": resume_skills_list, # For display/highlighting
            "resume_locations": resume_locations_list # For display
        }
        matched_resumes.append(match_details)

    # Sort resumes by final_match_score in descending order
    matched_resumes.sort(key=lambda x: x["final_match_score"], reverse=True)

    return matched_resumes

if __name__ == '__main__':
    print("Running matcher.py directly for testing...")

    # Dummy data for testing (mimicking outputs of previous modules)
    sample_parsed_resumes = [
        {
            "filename": "resume1.pdf", "name": "John Doe",
            "locations": ["New York, NY", "San Francisco, CA"],
            "skills": ["python", "machine learning", "django", "fastapi"],
            "cleaned_text": "python developer with machine learning skills and django."
        },
        {
            "filename": "resume2.docx", "name": "Jane Smith",
            "locations": ["Chicago, IL"],
            "skills": ["java", "aws", "spring", "kubernetes"],
            "cleaned_text": "java engineer skilled in aws and spring."
        },
        {
            "filename": "resume3.pdf", "name": "Alex Lee",
            "locations": ["Remote", "Austin, TX"],
            "skills": ["javascript", "react", "node.js"],
            "cleaned_text": "frontend developer with react experience."
        },
        {
            "filename": "resume4.pdf", "name": "Maria Garcia",
            "locations": ["New York, NY"],
            "skills": ["python", "fastapi", "docker"],
            "cleaned_text": "Experienced Python and FastAPI developer. Based in New York."
        }
    ]

    # Dummy embeddings (replace with actual embeddings if running integrated)
    # For 'all-MiniLM-L6-v2', dimension is 384
    EMBED_DIM = 384
    sample_resume_embeddings = {
        "resume1.pdf": np.random.rand(EMBED_DIM),
        "resume2.docx": np.random.rand(EMBED_DIM),
        "resume3.pdf": np.random.rand(EMBED_DIM),
        "resume4.pdf": np.random.rand(EMBED_DIM)
    }
    # Make resume1 and resume4 more similar to the job
    sample_resume_embeddings["resume1.pdf"] = np.array([0.1] * EMBED_DIM)
    sample_resume_embeddings["resume4.pdf"] = np.array([0.12] * EMBED_DIM) # Very similar to resume1
    sample_resume_embeddings["resume2.docx"] = np.array([0.5] * EMBED_DIM) # Different
    sample_resume_embeddings["resume3.pdf"] = np.array([0.8] * EMBED_DIM) # Very different


    sample_job_details = {
        "id": "job123", "title": "Senior Python Developer",
        "description": "Looking for a python developer with fastapi experience.",
        "location": "New York, NY",
        "skills": ["python", "fastapi", "machine learning", "docker"]
    }
    sample_job_embedding = np.array([0.11] * EMBED_DIM) # Similar to resume1 and resume4

    print("\n--- Test: Cosine Similarity ---")
    sim = calculate_cosine_similarity(sample_resume_embeddings["resume1.pdf"], sample_job_embedding)
    print(f"Similarity (resume1 vs job): {sim:.4f} (expected high)")
    sim_diff = calculate_cosine_similarity(sample_resume_embeddings["resume3.pdf"], sample_job_embedding)
    print(f"Similarity (resume3 vs job): {sim_diff:.4f} (expected low)")

    print("\n--- Test: Location Compatibility ---")
    print(f"Resume1 ({sample_parsed_resumes[0]['locations']}) vs Job ({sample_job_details['location']}): {check_location_compatibility(sample_parsed_resumes[0]['locations'], sample_job_details['location'])} (expected True)")
    print(f"Resume2 ({sample_parsed_resumes[1]['locations']}) vs Job ({sample_job_details['location']}): {check_location_compatibility(sample_parsed_resumes[1]['locations'], sample_job_details['location'])} (expected False)")

    print("\n--- Test: Skill Overlap Score ---")
    overlap1 = calculate_skill_overlap_score(sample_parsed_resumes[0]['skills'], sample_job_details['skills'])
    print(f"Skill overlap (resume1 vs job): {overlap1:.4f} (Expected: 3/4 = 0.75 for ['python', 'fastapi', 'machine learning'])")
    # Resume1 skills: ["python", "machine learning", "django", "fastapi"]
    # Job skills: ["python", "fastapi", "machine learning", "docker"]
    # Common: python, machine learning, fastapi (3). Job skills count: 4. Score: 3/4 = 0.75

    overlap2 = calculate_skill_overlap_score(sample_parsed_resumes[1]['skills'], sample_job_details['skills'])
    print(f"Skill overlap (resume2 vs job): {overlap2:.4f} (Expected: 0.0)")


    print("\n--- Test: Full Ranking ---")
    ranked_list = rank_resumes(
        sample_parsed_resumes,
        sample_resume_embeddings,
        sample_job_details,
        sample_job_embedding
    )

    print("Ranked Resumes:")
    for resume in ranked_list:
        print(
            f"  Filename: {resume['filename']}, Name: {resume['candidate_name']}, "
            f"Final Score: {resume['final_match_score']:.4f} "
            f"(Sem: {resume['semantic_similarity_score']:.4f}, "
            f"Loc: {resume['location_match']}, "
            f"Skill: {resume['skill_overlap_score']:.4f})"
        )

    # Check if resume4 or resume1 is at the top due to high similarity, location and skill match
    if ranked_list and (ranked_list[0]['filename'] == 'resume4.pdf' or ranked_list[0]['filename'] == 'resume1.pdf'):
        print("Ranking seems plausible (resume4 or resume1 is top).")
    else:
        print("Ranking top result might be unexpected. Check logic.")

    print("\nMatcher test finished.")
