import os
import fitz  # PyMuPDF
import docx2txt
from typing import List, Dict, Union

def extract_text_from_pdf(file_path: str) -> Union[str, None]:
    """Extracts text from a PDF file."""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {file_path}: {e}")
        return None

def extract_text_from_docx(file_path: str) -> Union[str, None]:
    """Extracts text from a DOCX file."""
    try:
        text = docx2txt.process(file_path)
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX {file_path}: {e}")
        return None

def load_resumes_from_directory(directory_path: str = "resumes/") -> List[Dict[str, str]]:
    """
    Loads all resumes (PDF and DOCX) from the specified directory.

    Args:
        directory_path (str): The path to the directory containing resumes.
                              Assumed to be relative to the project root (resume_matcher/).

    Returns:
        List[Dict[str, str]]: A list of dictionaries, where each dictionary
                               contains 'filename' and 'text' of the resume.
    """
    resumes_data = []
    # Adjust path to be relative to the script's location if necessary,
    # or ensure the calling context handles the path correctly.
    # For now, assuming 'directory_path' is correctly pointing to the resumes folder.

    # Construct absolute path to resumes directory relative to this script file
    script_dir = os.path.dirname(os.path.abspath(__file__)) # directory of data_ingestion.py
    actual_resumes_dir = os.path.join(script_dir, directory_path)

    if not os.path.isdir(actual_resumes_dir):
        print(f"Directory not found: {actual_resumes_dir}")
        # Create the directory if it doesn't exist, as it's expected.
        os.makedirs(actual_resumes_dir, exist_ok=True)
        print(f"Created directory: {actual_resumes_dir}")
        return [] # Return empty list if directory was missing (it will be empty)

    for filename in os.listdir(actual_resumes_dir):
        file_path = os.path.join(actual_resumes_dir, filename)
        text = None
        if filename.lower().endswith(".pdf"):
            print(f"Processing PDF: {file_path}")
            text = extract_text_from_pdf(file_path)
        elif filename.lower().endswith(".docx"):
            print(f"Processing DOCX: {file_path}")
            text = extract_text_from_docx(file_path)

        if text:
            resumes_data.append({"filename": filename, "text": text})
        elif text is None and (filename.lower().endswith(".pdf") or filename.lower().endswith(".docx")):
            # If text extraction failed but it was a target filetype, record it with empty text
            # or handle as an error case more explicitly if preferred.
            resumes_data.append({"filename": filename, "text": ""}) # Or perhaps skip adding it

    return resumes_data

if __name__ == '__main__':
    # Create dummy resume files for testing
    # Note: This part of the script will run when executed directly (e.g., python data_ingestion.py)
    # It's good for simple testing but for a real app, tests should be separate.

    print("Running data_ingestion.py directly for testing...")

    # Ensure the resumes directory exists relative to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_resumes_dir = os.path.join(script_dir, "resumes/") # Path to resume_matcher/resumes/
    os.makedirs(test_resumes_dir, exist_ok=True)

    # Create a dummy PDF file
    dummy_pdf_path = os.path.join(test_resumes_dir, "dummy_resume.pdf")
    try:
        doc = fitz.open() # Create a new empty PDF
        page = doc.new_page()
        page.insert_text((50, 72), "This is a dummy PDF resume for John Doe. Skills: Python, FastAPI.")
        doc.save(dummy_pdf_path)
        doc.close()
        print(f"Created dummy PDF: {dummy_pdf_path}")
    except Exception as e:
        print(f"Could not create dummy PDF: {e}")

    # Create a dummy DOCX file (using simple text for this example as python-docx is not installed by default)
    # For a real DOCX, you'd use a library like python-docx to create it.
    # Here we'll just write a text file and name it .docx for simplicity of the example,
    # knowing docx2txt might not process it if it's not a valid DOCX format.
    # A better approach for testing would be to have actual small dummy docx files.
    dummy_docx_path = os.path.join(test_resumes_dir, "dummy_resume.docx")
    try:
        # For this test to work with docx2txt, a real (even if simple) docx is needed.
        # Creating a simple text file and renaming it won't work.
        # Let's assume we have a tiny valid docx file for testing or skip this if python-docx is not to be added.
        # For now, we'll just print a message that a real docx should be there.
        if not os.path.exists(dummy_docx_path):
             print(f"Please place a real dummy DOCX file at: {dummy_docx_path} for testing docx extraction.")
        else:
            print(f"Dummy DOCX found: {dummy_docx_path}")

    except Exception as e:
        print(f"Error related to dummy DOCX: {e}")

    print(f"Attempting to load resumes from: {test_resumes_dir}")
    extracted_data = load_resumes_from_directory(directory_path="resumes/") # Relative path from data_ingestion.py

    if extracted_data:
        for item in extracted_data:
            print(f"--- Loaded {item['filename']} ---")
            # print(item['text'][:200] + "..." if item['text'] else "No text extracted.") # Print snippet
            print(f"Preview: {item['text'][:100] if item['text'] else 'Text extraction failed or file empty.'}")
    else:
        print("No resumes were loaded. Check paths and dummy files.")

    print("Data ingestion test finished.")
