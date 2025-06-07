import streamlit as st
import requests
import time
import os
from typing import List, Dict, Any, Optional

# Configuration
# Assuming FastAPI backend is running on localhost:8000
# If your project structure means api.py is run from `resume_matcher/`
# and streamlit_app.py is in `resume_matcher/frontend_ui/`,
# then the API is at http://localhost:8000
# Plots are served by FastAPI from /generated_plots/<filename>
# or Streamlit can access them if it has filesystem access to that dir.
# Let's assume FastAPI serves them.

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
POST_JOB_ENDPOINT = f"{BACKEND_URL}/post-job"
GET_MATCHES_ENDPOINT = f"{BACKEND_URL}/get-matches"
# Path from where FastAPI serves plots (e.g., /generated_plots/plot_name.png)
# This should match the 'name' argument in app.mount in api.py
PLOTS_ENDPOINT_BASE = f"{BACKEND_URL}/generated_plots"

# --- Helper Functions ---
def post_job_to_backend(title: str, description: str, skills: List[str], location: Optional[str], salary: Optional[str]) -> Optional[Dict[str, Any]]:
    """Sends job details to the backend API."""
    payload = {
        "title": title,
        "description": description,
        "skills": skills,
        "location": location,
        "salary": salary
    }
    try:
        response = requests.post(POST_JOB_ENDPOINT, json=payload, timeout=10) # 10s timeout for initial request
        response.raise_for_status() # Raise an exception for HTTP errors
        return response.json() # {"job_id": ..., "message": ..., "status_url": ...}
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")
        return None

def get_job_results_from_backend(job_id: str) -> Optional[Dict[str, Any]]:
    """Polls the backend for job results."""
    try:
        response = requests.get(f"{GET_MATCHES_ENDPOINT}/{job_id}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        # This can happen if the server is slow to respond initially or during polling
        # st.warning(f"Error fetching results (will retry): {e}")
        return None


# --- Streamlit UI ---
st.set_page_config(page_title="Resume Matcher", layout="wide")
st.title("🎯 Resume Matcher for HR")
st.markdown("Enter job requirements below to find the best candidate matches from the resume pool.")

# --- Job Input Form ---
with st.sidebar:
    st.header("📋 Job Requirements")
    job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer")
    job_description = st.text_area("Job Description", height=200, placeholder="e.g., Design, develop, and maintain software solutions...")
    job_skills_str = st.text_input("Required Skills (comma-separated)", placeholder="e.g., Python, FastAPI, Machine Learning")
    job_location = st.text_input("Location (Optional)", placeholder="e.g., New York, NY or Remote")
    job_salary = st.text_input("Salary Range (Optional)", placeholder="e.g., $100,000 - $120,000")

    submit_button = st.button("🚀 Find Matches")

# Session state to store job ID and results
if 'job_id' not in st.session_state:
    st.session_state.job_id = None
if 'job_results' not in st.session_state:
    st.session_state.job_results = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False


if submit_button:
    if not job_title or not job_description or not job_skills_str:
        st.warning("Please fill in Job Title, Description, and Skills.")
    else:
        st.session_state.job_id = None # Reset previous job
        st.session_state.job_results = None
        st.session_state.error_message = None
        st.session_state.processing_complete = False

        skills_list = [skill.strip() for skill in job_skills_str.split(',') if skill.strip()]

        with st.spinner("Submitting job to backend..."):
            response_data = post_job_to_backend(job_title, job_description, skills_list, job_location, job_salary)

        if response_data and response_data.get("job_id"):
            st.session_state.job_id = response_data["job_id"]
            st.success(f"✅ Job submitted successfully! Job ID: {st.session_state.job_id}. Waiting for results...")
            st.info("The system is now processing resumes. This might take a moment. Results will appear below.")
        else:
            st.error("Failed to submit job. Please check backend connection or input.")
            st.session_state.error_message = "Failed to submit job to the backend."


# --- Polling and Displaying Results ---
if st.session_state.job_id and not st.session_state.processing_complete:
    with st.spinner(f"⏳ Waiting for results for Job ID: {st.session_state.job_id}... (This may take a minute or two for the first run)"):
        POLL_INTERVAL = 5  # seconds
        MAX_ATTEMPTS = 30 # Max 2.5 minutes of polling for this example

        for attempt in range(MAX_ATTEMPTS):
            results = get_job_results_from_backend(st.session_state.job_id)
            if results:
                st.session_state.job_results = results
                if results.get("status") == "COMPLETED":
                    st.success("🎉 Processing complete! Results below.")
                    st.session_state.processing_complete = True
                    st.session_state.error_message = None
                    break
                elif results.get("status") == "FAILED":
                    st.error(f"Backend processing failed: {results.get('error_message', 'Unknown error')}")
                    st.session_state.error_message = results.get('error_message', 'Unknown error')
                    st.session_state.processing_complete = True # Stop polling on failure
                    break
                elif results.get("status") == "PROCESSING":
                    st.info(f"Still processing... Attempt {attempt + 1}/{MAX_ATTEMPTS}")
                # If PENDING or other status, continue polling
            else: # results is None, likely a connection error during polling
                 st.warning(f"Could not fetch results on attempt {attempt + 1}. Will retry...")

            time.sleep(POLL_INTERVAL)

        if not st.session_state.processing_complete and not st.session_state.error_message:
            # If loop finishes without completion or specific failure message
            st.warning("Polling timed out. The backend might still be processing or encountered an issue. Please check backend logs or try again later.")
            st.session_state.error_message = "Polling timed out."


# Display results if available and processing is marked complete
if st.session_state.processing_complete and st.session_state.job_results:
    results_data = st.session_state.job_results

    if results_data.get("status") == "COMPLETED":
        st.header(f"📊 Match Results for Job: {results_data.get('job_details', {}).get('title', st.session_state.job_id)}")

        ranked_matches = results_data.get("ranked_matches", [])
        visualizations = results_data.get("visualizations", {})

        if not ranked_matches:
            st.info("No matching candidates found based on the criteria.")
        else:
            st.subheader(f"🏆 Top {len(ranked_matches)} Matched Candidates")

            # Use columns for a more compact display
            cols = st.columns(3)
            col_names = ["Candidate", "Overall Score (%)", "Details"]
            for i, col_name in enumerate(col_names):
                cols[i].markdown(f"**{col_name}**")

            for i, match in enumerate(ranked_matches):
                cols = st.columns(3)
                cols[0].write(match.get('candidate_name', match.get('filename', 'N/A')))
                cols[1].write(f"{match.get('final_match_score', 0) * 100:.2f}%")
                with cols[2].expander("View Details"):
                    st.markdown(f"**File:** {match.get('filename', 'N/A')}")
                    st.markdown(f"**Semantic Similarity:** {match.get('semantic_similarity_score', 0)*100:.2f}%")
                    st.markdown(f"**Location Match:** {'✅ Yes' if match.get('location_match') else '❌ No'}")
                    st.markdown(f"**Skill Overlap:** {match.get('skill_overlap_score', 0)*100:.2f}%")
                    st.markdown(f"**Identified Skills:** {', '.join(match.get('resume_skills', [])) if match.get('resume_skills') else 'None'}")
                    st.markdown(f"**Identified Locations:** {', '.join(match.get('resume_locations', [])) if match.get('resume_locations') else 'None'}")

            st.markdown("---")
            st.subheader("📈 Visual Insights")

            # Display plots (assuming FastAPI serves them from /generated_plots/<filename>)
            # The visualization module saves plots with job_id in filename.
            # The API returns just the basename of the plot.

            vis_cols = st.columns(len(visualizations) if visualizations else 1)
            plot_idx = 0
            if visualizations.get("top_candidate_scores_plot"):
                plot_url = f"{PLOTS_ENDPOINT_BASE}/{visualizations['top_candidate_scores_plot']}"
                with vis_cols[plot_idx % len(vis_cols)]: # Distribute plots into columns
                    st.image(plot_url, caption="Top Candidate Scores", use_column_width=True)
                plot_idx +=1

            if visualizations.get("skills_heatmap_plot"):
                plot_url = f"{PLOTS_ENDPOINT_BASE}/{visualizations['skills_heatmap_plot']}"
                with vis_cols[plot_idx % len(vis_cols)]:
                    st.image(plot_url, caption="Skills Match Heatmap", use_column_width=True)
                plot_idx +=1

            if visualizations.get("similarity_distribution_plot"):
                plot_url = f"{PLOTS_ENDPOINT_BASE}/{visualizations['similarity_distribution_plot']}"
                with vis_cols[plot_idx % len(vis_cols)]:
                    st.image(plot_url, caption="Semantic Similarity Distribution", use_column_width=True)
                plot_idx +=1

            if not visualizations:
                st.info("No visualizations were generated for this job.")

    elif results_data.get("status") == "FAILED":
        # This case is already handled by the polling logic's st.error, but good for completeness
        if not st.session_state.error_message: # If not already shown by poller
            st.error(f"Job processing failed: {results_data.get('error_message', 'Unknown error')}")

elif st.session_state.error_message and st.session_state.processing_complete:
    # If polling stopped due to an error message that wasn't a FAILED status from backend (e.g. timeout)
    st.error(f"Could not retrieve results: {st.session_state.error_message}")

st.sidebar.markdown("---")
st.sidebar.info("This UI interacts with a FastAPI backend to process resumes and find matches.")
st.sidebar.markdown("Ensure the backend server is running.")

# To run this Streamlit app:
# Ensure you are in the directory containing `resume_matcher/`
# Then run: streamlit run resume_matcher/frontend_ui/streamlit_app.py
