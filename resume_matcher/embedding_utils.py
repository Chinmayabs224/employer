from sentence_transformers import SentenceTransformer
import numpy as np
import os
import joblib # For saving/loading embeddings
from typing import List, Union, Dict, Any

# Load a pre-trained Sentence-BERT model
# This model will be downloaded automatically by sentence-transformers on first use.
# Using a relatively small but effective model.
# Other models like 'all-mpnet-base-v2' are larger and potentially more accurate but slower.
MODEL_NAME = 'all-MiniLM-L6-v2'
try:
    model = SentenceTransformer(MODEL_NAME)
except Exception as e:
    print(f"Error loading SentenceTransformer model {MODEL_NAME}: {e}")
    # Potentially handle proxy or network issues if that's a common problem in the environment
    # For now, just re-raise or exit if the model can't be loaded.
    raise

# Define a path for caching embeddings
EMBEDDINGS_CACHE_DIR = "embeddings_cache"
os.makedirs(EMBEDDINGS_CACHE_DIR, exist_ok=True)

def get_embedding(text: Union[str, List[str]]) -> np.ndarray:
    """
    Generates an embedding for a given text or list of texts using Sentence-BERT.
    """
    if not text:
        # Return a zero vector or handle as an error if text is empty.
        # The dimensionality should match the model's output.
        # For all-MiniLM-L6-v2, it's 384.
        return np.zeros(model.get_sentence_embedding_dimension())

    # Ensure text is a list for the encode function for consistency,
    # even if it's a single string.
    if isinstance(text, str):
        # If it's a single string that might be very long, consider splitting it
        # or ensuring the model can handle its length. Most SBERT models have input limits.
        # For now, assuming single strings are manageable.
        embeddings = model.encode([text], convert_to_numpy=True)
        return embeddings[0] # Return the single embedding
    else: # It's a list of strings
        embeddings = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embeddings # Return the array of embeddings


def generate_resume_embeddings(parsed_resumes: List[Dict[str, Any]], force_regenerate: bool = False) -> Dict[str, np.ndarray]:
    """
    Generates or loads embeddings for key sections of parsed resumes.
    For simplicity, let's focus on embedding the 'cleaned_text' of the resume.
    More advanced: embed sections like projects, skills summary, etc., separately or concatenated.

    Args:
        parsed_resumes (List[Dict[str, Any]]): List of resume data,
                                               output from resume_parser.py.
                                               Each dict must have 'filename' and 'cleaned_text'.
        force_regenerate (bool): If True, ignore cached embeddings and regenerate all.


    Returns:
        Dict[str, np.ndarray]: A dictionary mapping resume filenames to their embeddings.
    """
    resume_embeddings = {}
    cache_file_path = os.path.join(EMBEDDINGS_CACHE_DIR, "resume_embeddings_cache.joblib")

    if not force_regenerate and os.path.exists(cache_file_path):
        try:
            resume_embeddings = joblib.load(cache_file_path)
            print(f"Loaded resume embeddings from cache: {cache_file_path}")
        except Exception as e:
            print(f"Error loading resume embeddings from cache: {e}. Regenerating.")
            resume_embeddings = {} # Reset if cache is corrupted

    for resume_data in parsed_resumes:
        filename = resume_data.get("filename")
        if not filename:
            print("Warning: Parsed resume data missing filename. Skipping.")
            continue

        if not force_regenerate and filename in resume_embeddings:
            # print(f"Using cached embedding for {filename}")
            continue # Already loaded from cache

        print(f"Generating embedding for resume: {filename}")
        # We'll embed the 'cleaned_text' which should be present from the parser.
        text_to_embed = resume_data.get("cleaned_text", "")
        if not text_to_embed:
            print(f"Warning: No 'cleaned_text' found for {filename}. Using empty string.")

        embedding = get_embedding(text_to_embed)
        resume_embeddings[filename] = embedding

    try:
        joblib.dump(resume_embeddings, cache_file_path)
        print(f"Saved/Updated resume embeddings cache: {cache_file_path}")
    except Exception as e:
        print(f"Error saving resume embeddings to cache: {e}")

    return resume_embeddings


def generate_job_description_embedding(job_details: Dict[str, Any], force_regenerate: bool = False) -> Union[np.ndarray, None]:
    """
    Generates or loads an embedding for a job description.
    The job_details dictionary should contain fields like 'title', 'description', 'skills'.
    We can concatenate these to form a representative text for embedding.

    Args:
        job_details (Dict[str, Any]): Dictionary containing job details.
                                      Must include 'description', 'title'. 'skills' are optional.
        force_regenerate (bool): If True, ignore cached embedding and regenerate.

    Returns:
        np.ndarray: The embedding for the job description.
    """
    job_id = job_details.get("id", job_details.get("title", "unknown_job")) # Create a unique ID for caching
    cache_file_path = os.path.join(EMBEDDINGS_CACHE_DIR, f"job_{job_id}_embedding.joblib")

    if not force_regenerate and os.path.exists(cache_file_path):
        try:
            embedding = joblib.load(cache_file_path)
            print(f"Loaded job description embedding from cache for: {job_id}")
            return embedding
        except Exception as e:
            print(f"Error loading job description embedding from cache for {job_id}: {e}. Regenerating.")

    print(f"Generating embedding for job: {job_id}")
    title = job_details.get("title", "")
    description = job_details.get("description", "")
    skills = job_details.get("skills", []) # skills might be a list

    # Concatenate relevant fields to create a comprehensive text for embedding
    if isinstance(skills, list):
        skills_text = ", ".join(skills)
    else: # skills might be a string
        skills_text = skills

    text_to_embed = f"Job Title: {title}. Description: {description}. Required Skills: {skills_text}."

    if not text_to_embed.strip(". Job Title: . Description: . Required Skills: ."): # Check if it's effectively empty
        print(f"Warning: Not enough information to generate embedding for job: {job_id}")
        return None

    embedding = get_embedding(text_to_embed)

    try:
        joblib.dump(embedding, cache_file_path)
        print(f"Saved job description embedding to cache for: {job_id}")
    except Exception as e:
        print(f"Error saving job description embedding to cache for {job_id}: {e}")

    return embedding

if __name__ == '__main__':
    print("Running embedding_utils.py directly for testing...")

    # Dummy parsed resume data (replace with actual output from resume_parser if available)
    sample_parsed_resumes = [
        {
            "filename": "resume1.pdf",
            "cleaned_text": "Experienced python developer with skills in machine learning and web development. Worked on several projects using Django and Scikit-learn.",
            "name": "John Doe",
            "skills": ["python", "machine learning", "django"]
        },
        {
            "filename": "resume2.docx",
            "cleaned_text": "Software engineer specializing in Java and cloud computing. Proficient with AWS services and Spring framework. Good java programmer.",
            "name": "Jane Smith",
            "skills": ["java", "aws", "spring"]
        },
        { # A resume with minimal text
            "filename": "resume3.pdf",
            "cleaned_text": "Entry level.",
            "name": "New Grad",
            "skills": []
        }
    ]

    print("\n--- Testing Resume Embeddings ---")
    # Test with force_regenerate=True first to ensure generation works
    print("Generating with force_regenerate=True:")
    resume_embeds = generate_resume_embeddings(sample_parsed_resumes, force_regenerate=True)
    for filename, embed in resume_embeds.items():
        print(f"Embedding for {filename} - Shape: {embed.shape}, Non-zero: {np.any(embed)}")

    # Test loading from cache
    print("\nGenerating again (should load from cache):")
    resume_embeds_cached = generate_resume_embeddings(sample_parsed_resumes, force_regenerate=False)
    if "resume1.pdf" in resume_embeds_cached and np.array_equal(resume_embeds["resume1.pdf"], resume_embeds_cached["resume1.pdf"]):
        print("Resume embeddings seem to be cached and loaded correctly.")
    else:
        print("Issue with resume embedding caching or loading.")


    print("\n--- Testing Job Description Embedding ---")
    sample_job_details = {
        "id": "job123",
        "title": "Senior Python Developer",
        "description": "We are looking for a senior python developer with experience in FastAPI and machine learning applications. The ideal candidate will lead a team and develop new features.",
        "skills": ["python", "fastapi", "machine learning", "leadership"]
    }

    # Test with force_regenerate=True
    print("Generating job embedding with force_regenerate=True:")
    job_embed = generate_job_description_embedding(sample_job_details, force_regenerate=True)
    if job_embed is not None:
        print(f"Job embedding for {sample_job_details['id']} - Shape: {job_embed.shape}, Non-zero: {np.any(job_embed)}")

    # Test loading from cache
    print("\nGenerating job embedding again (should load from cache):")
    job_embed_cached = generate_job_description_embedding(sample_job_details, force_regenerate=False)
    if job_embed_cached is not None and np.array_equal(job_embed, job_embed_cached):
        print("Job description embedding seems to be cached and loaded correctly.")
    else:
        print("Issue with job description embedding caching or loading.")

    # Test with minimal job details
    minimal_job = {"id": "job000", "title": "Intern", "description": "Basic tasks."}
    print("\nGenerating job embedding for minimal job description:")
    minimal_job_embed = generate_job_description_embedding(minimal_job, force_regenerate=True)
    if minimal_job_embed is not None:
        print(f"Job embedding for {minimal_job['id']} - Shape: {minimal_job_embed.shape}")

    # Test with empty job details (should return None or zero vector)
    empty_job = {"id": "jobempty"}
    print("\nGenerating job embedding for empty job description:")
    empty_job_embed = generate_job_description_embedding(empty_job, force_regenerate=True)
    if empty_job_embed is None or not np.any(empty_job_embed): # Check if it's None or all zeros
        print("Handled empty job description appropriately (None or zero vector).")
    else:
        print(f"Empty job description resulted in embedding: {empty_job_embed.shape}")

    print("\nEmbedding utils test finished.")
