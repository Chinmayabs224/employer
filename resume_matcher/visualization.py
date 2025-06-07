import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting (important for servers/scripts)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union
import os

# Define a directory to save plots
PLOTS_DIR = "generated_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_top_candidate_scores(
    matched_resumes: List[Dict[str, Any]],
    top_n: int = 10,
    job_id: str = "job"
) -> Union[str, None]:
    """
    Generates a horizontal bar chart of final match scores for the top N candidates.
    Saves the plot as a PNG file.

    Args:
        matched_resumes: List of matched resume dicts from matcher.py.
                         Each dict needs 'candidate_name' and 'final_match_score'.
        top_n: Number of top candidates to display.
        job_id: An identifier for the job, used in the filename.

    Returns:
        str: Path to the saved plot image, or None if plotting failed.
    """
    if not matched_resumes:
        print("No matched resumes to plot.")
        return None

    # Take top N candidates (already sorted by matcher)
    top_candidates = matched_resumes[:top_n]

    candidate_names = [c.get('candidate_name', f"Resume_{i}") for i, c in enumerate(top_candidates)]
    scores = [c.get('final_match_score', 0) * 100 for c in top_candidates] # Convert to percentage

    if not candidate_names or not scores:
        print("Not enough data for plotting top candidate scores.")
        return None

    plt.figure(figsize=(10, max(6, len(candidate_names) * 0.5))) # Adjust height based on N
    sns.set_style("whitegrid")

    bars = plt.barh(candidate_names, scores, color=sns.color_palette("viridis", len(candidate_names)))

    plt.xlabel("Overall Match Score (%)")
    plt.ylabel("Candidate")
    plt.title(f"Top {len(candidate_names)} Candidates - Match Scores")
    plt.xlim(0, 100) # Scores are 0-1, scaled to 0-100

    # Add score labels on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2., f'{width:.2f}%',
                 ha='left', va='center')

    plt.gca().invert_yaxis() # Display top candidate at the top
    plt.tight_layout()

    plot_filename = f"{job_id}_top_candidates_scores.png"
    plot_path = os.path.join(PLOTS_DIR, plot_filename)

    try:
        plt.savefig(plot_path)
        plt.close() # Close the figure to free memory
        print(f"Saved top candidate scores plot to: {plot_path}")
        return plot_path
    except Exception as e:
        print(f"Error saving plot: {e}")
        plt.close()
        return None


def plot_skills_heatmap_table(
    matched_resumes: List[Dict[str, Any]],
    job_skills: List[str],
    top_n: int = 5,
    job_id: str = "job"
) -> Union[str, None]:
    """
    Generates a heatmap-like table showing skill presence for top N candidates against job skills.
    Saves the plot as a PNG file.

    Args:
        matched_resumes: List of matched resume dicts. Needs 'candidate_name' and 'resume_skills'.
        job_skills: List of skills required for the job.
        top_n: Number of top candidates to display.
        job_id: An identifier for the job, used in the filename.

    Returns:
        str: Path to the saved plot image, or None if plotting failed.
    """
    if not matched_resumes or not job_skills:
        print("No matched resumes or job skills to plot for heatmap.")
        return None

    top_candidates = matched_resumes[:top_n]
    if not top_candidates:
        print("No candidates to display in skills heatmap.")
        return None

    candidate_names = [c.get('candidate_name', f"Resume_{i}") for i,c in enumerate(top_candidates)]

    # Normalize job skills for consistent checking
    norm_job_skills = [skill.lower().strip() for skill in job_skills]

    # Create a DataFrame for the heatmap data: 1 if skill present, 0 otherwise
    heatmap_data = []
    for candidate in top_candidates:
        resume_skills_norm = [skill.lower().strip() for skill in candidate.get('resume_skills', [])]
        skill_presence = [1 if job_skill in resume_skills_norm else 0 for job_skill in norm_job_skills]
        heatmap_data.append(skill_presence)

    if not heatmap_data:
        print("Could not generate data for skills heatmap.")
        return None

    df_heatmap = pd.DataFrame(heatmap_data, index=candidate_names, columns=norm_job_skills)

    plt.figure(figsize=(max(8, len(norm_job_skills) * 0.8), max(4, len(candidate_names) * 0.5)))
    sns.set_theme(style="white") # Use a seaborn theme

    # Using a discrete colormap (e.g., just two colors for 0 and 1)
    cmap = sns.color_palette(["#f0f0f0", "#2ecc71"], as_cmap=True) # Light grey for absent, green for present

    ax = sns.heatmap(df_heatmap, annot=True, cmap=cmap, cbar=False, linewidths=.5, linecolor='gray', fmt="d")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title(f"Skills Match for Top {len(candidate_names)} Candidates")
    plt.ylabel("Candidate")
    plt.xlabel("Required Job Skills")
    plt.tight_layout()

    plot_filename = f"{job_id}_skills_heatmap.png"
    plot_path = os.path.join(PLOTS_DIR, plot_filename)

    try:
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved skills heatmap plot to: {plot_path}")
        return plot_path
    except Exception as e:
        print(f"Error saving skills heatmap plot: {e}")
        plt.close()
        return None


def plot_semantic_similarity_distribution(
    all_scores: List[float], # List of semantic similarity scores for all candidates
    job_id: str = "job"
) -> Union[str, None]:
    """
    Generates a histogram of semantic similarity scores for all candidates.
    Saves the plot as a PNG file.

    Args:
        all_scores: List of semantic similarity scores (floats between -1 and 1, typically 0 to 1).
        job_id: An identifier for the job, used in the filename.

    Returns:
        str: Path to the saved plot image, or None if plotting failed.
    """
    if not all_scores:
        print("No similarity scores to plot for distribution.")
        return None

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Scores are typically between 0 and 1 after SBERT for similarity tasks.
    # Filter out potential outliers if any, or ensure scores are in expected range.
    scores_to_plot = [s for s in all_scores if 0 <= s <= 1]
    if not scores_to_plot: # if filtering removed all scores
        print("No valid scores in the range [0,1] to plot for distribution.")
        return None

    sns.histplot(scores_to_plot, bins=20, kde=True, color="skyblue")

    plt.xlabel("Semantic Similarity Score with Job Description")
    plt.ylabel("Number of Resumes")
    plt.title(f"Distribution of Semantic Similarity Scores (Job: {job_id})")
    plt.xlim(0, 1) # SBERT similarity scores are often in this range
    plt.tight_layout()

    plot_filename = f"{job_id}_similarity_distribution.png"
    plot_path = os.path.join(PLOTS_DIR, plot_filename)

    try:
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved similarity distribution plot to: {plot_path}")
        return plot_path
    except Exception as e:
        print(f"Error saving similarity distribution plot: {e}")
        plt.close()
        return None


if __name__ == '__main__':
    print("Running visualization.py directly for testing...")
    os.makedirs(PLOTS_DIR, exist_ok=True) # Ensure dir exists for tests

    # Dummy matched resumes data (from matcher.py)
    sample_matched_resumes = [
        {'filename': 'resume4.pdf', 'candidate_name': 'Maria Garcia',
         'semantic_similarity_score': 0.85, 'location_match': True, 'skill_overlap_score': 0.75,
         'final_match_score': 0.82, 'resume_skills': ['python', 'fastapi', 'docker'], 'resume_locations': ['New York, NY']},
        {'filename': 'resume1.pdf', 'candidate_name': 'John Doe',
         'semantic_similarity_score': 0.80, 'location_match': True, 'skill_overlap_score': 0.75,
         'final_match_score': 0.79, 'resume_skills': ['python', 'machine learning', 'django', 'fastapi'], 'resume_locations': ['New York, NY']},
        {'filename': 'resume_other.pdf', 'candidate_name': 'Sam Ray',
         'semantic_similarity_score': 0.60, 'location_match': False, 'skill_overlap_score': 0.25,
         'final_match_score': 0.475, 'resume_skills': ['java', 'testing'], 'resume_locations': ['Boston, MA']},
        {'filename': 'resume_another.pdf', 'candidate_name': 'Chris Lee',
         'semantic_similarity_score': 0.50, 'location_match': True, 'skill_overlap_score': 0.10,
         'final_match_score': 0.42, 'resume_skills': ['javascript'], 'resume_locations': ['New York, NY']},
    ]

    # Dummy job skills for heatmap
    sample_job_skills = ["python", "fastapi", "docker", "communication"]

    # Dummy all similarity scores for distribution plot
    all_semantic_scores = [0.85, 0.80, 0.60, 0.50, 0.45, 0.78, 0.30, 0.65, 0.77, 0.20]

    print("\n--- Test: Plot Top Candidate Scores ---")
    plot_path_scores = plot_top_candidate_scores(sample_matched_resumes, top_n=3, job_id="test_job1")
    if plot_path_scores and os.path.exists(plot_path_scores):
        print(f"Successfully generated top candidate scores plot: {plot_path_scores}")
    else:
        print(f"Failed to generate top candidate scores plot or file not found.")

    print("\n--- Test: Plot Skills Heatmap/Table ---")
    plot_path_heatmap = plot_skills_heatmap_table(sample_matched_resumes, sample_job_skills, top_n=3, job_id="test_job1")
    if plot_path_heatmap and os.path.exists(plot_path_heatmap):
        print(f"Successfully generated skills heatmap plot: {plot_path_heatmap}")
    else:
        print(f"Failed to generate skills heatmap plot or file not found.")

    print("\n--- Test: Plot Semantic Similarity Distribution ---")
    plot_path_dist = plot_semantic_similarity_distribution(all_semantic_scores, job_id="test_job1")
    if plot_path_dist and os.path.exists(plot_path_dist):
        print(f"Successfully generated similarity distribution plot: {plot_path_dist}")
    else:
        print(f"Failed to generate similarity distribution plot or file not found.")

    # Test with empty data
    print("\n--- Test: Plot with empty data ---")
    plot_empty_scores = plot_top_candidate_scores([], top_n=3, job_id="test_empty")
    if plot_empty_scores is None:
        print("Handled empty data correctly for top candidate scores.")

    plot_empty_heatmap = plot_skills_heatmap_table([], [], top_n=3, job_id="test_empty")
    if plot_empty_heatmap is None:
        print("Handled empty data correctly for skills heatmap.")

    plot_empty_dist = plot_semantic_similarity_distribution([], job_id="test_empty")
    if plot_empty_dist is None:
        print("Handled empty data correctly for similarity distribution.")

    print("\nVisualization test finished.")
