from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os

# Resume Matcher specific imports
from resume_matcher.embedding_utils import generate_job_description_embedding, generate_resume_embeddings, get_embedding
from resume_matcher.resume_parser import parse_resume_text
from resume_matcher.data_ingestion import load_resumes_from_directory, extract_text_from_pdf, extract_text_from_docx
from resume_matcher.matcher import rank_resumes, rank_jobs_for_resume # Updated import


app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configuration for upload folder
UPLOAD_FOLDER = 'uploads'
RESUMES_FOLDER_REL_PATH = os.path.join('resume_matcher', 'resumes') # Relative path for logic within app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Absolute path for saving files and ensuring directory exists from app.py's perspective
app.config['RESUMES_FOLDER_ABS'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESUMES_FOLDER_REL_PATH)

# Ensure upload and resume directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESUMES_FOLDER_ABS'], exist_ok=True)

# --- Global Data Stores (for simplicity) ---
jobs_data = {}
resumes_data = {}
resumes_loaded_and_processed = False

ALLOWED_EXTENSIONS_CSV = {'csv'}
ALLOWED_EXTENSIONS_RESUME = {'pdf', 'docx', 'txt'}

def extract_text_from_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error extracting text from TXT {file_path}: {e}")
        return None

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    return render_template('index.html')

# --- HR Routes ---
@app.route('/hr')
def hr_index():
    return render_template('hr_index.html', jobs_count=len(jobs_data))

@app.route('/hr/upload', methods=['GET', 'POST'])
def hr_upload_form():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_CSV):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                df = pd.read_csv(filepath)
                required_columns = ['job_id', 'title', 'description', 'skills']
                if not all(col in df.columns for col in required_columns):
                    flash(f'CSV must contain columns: {", ".join(required_columns)}', 'error')
                    return redirect(request.url)
                for index, row in df.iterrows():
                    job_id = str(row['job_id'])
                    job_details = {
                        'id': job_id,
                        'title': row['title'],
                        'description': row['description'],
                        'skills': row['skills'].split(',') if isinstance(row['skills'], str) else [],
                        'location': row.get('location', ''),
                        'salary': row.get('salary', '')
                    }
                    job_embedding = generate_job_description_embedding(job_details, force_regenerate=True)
                    if job_embedding is not None:
                        jobs_data[job_id] = {'details': job_details, 'embedding': job_embedding}
                    else:
                        flash(f'Could not generate embedding for job ID {job_id}. Skipping.', 'warning')
                flash(f'{len(df)} jobs processed from {filename}. Total jobs in store: {len(jobs_data)}', 'success')
                return redirect(url_for('hr_index'))
            except Exception as e:
                flash(f'Error processing CSV file: {e}', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload a CSV file.', 'error')
            return redirect(request.url)
    return render_template('hr_upload.html')

@app.route('/hr/matches')
def hr_view_matches():
    return render_template('hr_matches_overview.html', jobs=jobs_data)

@app.route('/hr/matches/<job_id>')
def hr_get_specific_job_matches(job_id):
    global resumes_data, resumes_loaded_and_processed
    job_info = jobs_data.get(job_id)
    if not job_info:
        flash(f'Job ID {job_id} not found.', 'error')
        return redirect(url_for('hr_view_matches'))
    if not resumes_loaded_and_processed:
        print("Loading and processing resumes for the first time (HR match view)...")
        raw_resumes = load_resumes_from_directory("resumes")
        parsed_resumes_list = []
        for r_data in raw_resumes:
            if r_data.get("text"):
                parsed_info = parse_resume_text(r_data["text"])
                parsed_info["filename"] = r_data.get("filename", os.path.basename(r_data.get("filepath", "Unknown")))
                parsed_resumes_list.append(parsed_info)
            else:
                print(f"Warning: No text extracted for {r_data.get('filename', 'Unknown file')}. Skipping.")
        if not parsed_resumes_list:
            flash("No resumes found or parsed in 'resume_matcher/resumes/'. Ensure it exists and contains resumes.", "warning")
        else:
            resume_embeddings_dict = generate_resume_embeddings(parsed_resumes_list, force_regenerate=False)
            for parsed_resume in parsed_resumes_list:
                fname = parsed_resume["filename"]
                if fname in resume_embeddings_dict:
                    resumes_data[fname] = {
                        "details": parsed_resume,
                        "embedding": resume_embeddings_dict[fname]
                    }
            resumes_loaded_and_processed = True
            flash(f'{len(resumes_data)} resumes loaded and processed.', 'info')
            print(f"Resumes loaded: {len(resumes_data)} parsed, embeddings generated.")
    if not resumes_data:
        flash("No resume data available. Upload resumes first.", "warning")
        return render_template('hr_specific_job_matches.html', job_id=job_id, job_details=job_info['details'], matches=[])
    current_parsed_resumes_list = [data['details'] for data in resumes_data.values()]
    current_resume_embeddings_dict = {fname: data['embedding'] for fname, data in resumes_data.items()}
    ranked_matches = rank_resumes(
        parsed_resumes_data=current_parsed_resumes_list,
        resume_embeddings=current_resume_embeddings_dict,
        job_details=job_info['details'],
        job_embedding=job_info['embedding']
    )
    return render_template('hr_specific_job_matches.html', job_id=job_id, job_details=job_info['details'], matches=ranked_matches)

# --- Candidate Routes ---

@app.route('/candidate')
def candidate_index():
    return render_template('candidate_index.html')

@app.route('/candidate/upload', methods=['GET', 'POST'])
def candidate_upload_form():
    global resumes_data
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_RESUME):
            filename = secure_filename(file.filename)
            resume_save_dir = app.config['RESUMES_FOLDER_ABS']
            filepath = os.path.join(resume_save_dir, filename)

            file.save(filepath)

            text = None
            if filename.lower().endswith(".pdf"):
                text = extract_text_from_pdf(filepath)
            elif filename.lower().endswith(".docx"):
                text = extract_text_from_docx(filepath)
            elif filename.lower().endswith(".txt"):
                text = extract_text_from_txt(filepath)

            if not text:
                flash(f'Could not extract text from {filename}.', 'error')
                return redirect(request.url)

            parsed_info = parse_resume_text(text)
            parsed_info["filename"] = filename

            resume_embedding = get_embedding(parsed_info.get("cleaned_text", ""))

            if resume_embedding is not None:
                resumes_data[filename] = {
                    "details": parsed_info,
                    "embedding": resume_embedding
                }
                flash(f'Resume {filename} uploaded and processed successfully.', 'success')
                global resumes_loaded_and_processed
                if not resumes_loaded_and_processed and resumes_data:
                     resumes_loaded_and_processed = True
                return redirect(url_for('candidate_view_matches', resume_filename=filename))
            else:
                flash(f'Could not generate embedding for {filename}.', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload a PDF, DOCX, or TXT file.', 'error')
            return redirect(request.url)

    return render_template('candidate_upload.html')

# rank_jobs_for_resume function was here, now removed. It's imported from resume_matcher.matcher

@app.route('/candidate/matches/<resume_filename>')
def candidate_view_matches(resume_filename):
    if resume_filename not in resumes_data:
        flash(f'Resume {resume_filename} not found or not processed.', 'error')
        return redirect(url_for('candidate_upload_form'))

    resume_info = resumes_data[resume_filename]

    if not jobs_data:
        flash('No job postings available to match against. Please ask HR to upload jobs.', 'warning')
        return render_template('candidate_matches.html', resume_filename=resume_filename, resume_details=resume_info['details'], matches=[])

    # Now calling the imported function
    matched_jobs = rank_jobs_for_resume(resume_info['details'], resume_info['embedding'], jobs_data)

    return render_template('candidate_matches.html', resume_filename=resume_filename, resume_details=resume_info['details'], matches=matched_jobs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
