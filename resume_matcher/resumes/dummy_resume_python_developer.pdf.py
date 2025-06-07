import fitz # PyMuPDF
import os

# Ensure the resumes directory exists
script_dir = os.path.dirname(os.path.abspath(__file__)) # This will be /app for create_file_with_block
resumes_dir = os.path.join(script_dir, "resume_matcher/resumes") # Adjust path based on actual execution context of this temp script
os.makedirs(resumes_dir, exist_ok=True)

file_path = os.path.join(resumes_dir, "dummy_resume_python_developer.pdf")
content = """Alice Wonderland - Python Developer
alice@example.com - New York, NY
Skills: Python, FastAPI, Django, SQL, Docker.
Project: Built a REST API with FastAPI for e-commerce.
Experience: 5 years as a backend developer."""

doc = fitz.open()
page = doc.new_page()
page.insert_text((50, 72), content) # Insert text at (horizontal, vertical) position
doc.save(file_path)
doc.close()
print(f"Created PDF: {file_path}")
