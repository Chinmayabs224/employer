import spacy
import re
from typing import Dict, List, Any, Union

# Load a spaCy model
# This will be downloaded by the subtask if not present.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model en_core_web_sm...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def clean_text(text: str) -> str:
    """Cleans the input text by lowercasing and removing excessive whitespace."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    # Add more cleaning rules as needed (e.g., remove special characters, URLs, emails)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'\S*@\S*\s?', '', text, flags=re.MULTILINE) # Remove email addresses
    return text

def extract_name(text: str) -> Union[str, None]:
    """Extracts person names from text using spaCy's NER."""
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Often names are in title case in resumes, even if we lowercase the whole text.
            # spaCy's NER is pretty good with this. We'll take the first one found.
            return ent.text.strip()
    return None

def extract_emails(text: str) -> List[str]:
    """Extracts email addresses from text using regex."""
    # Regex for finding email addresses
    email_regex = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.findall(email_regex, text)

def extract_phones(text: str) -> List[str]:
    """Extracts phone numbers from text using regex."""
    # Regex for finding phone numbers (supports various common formats)
    phone_regex = r'\+?\d{1,3}?[-.\s()]?\d{1,4}[-.\s()]?\d{1,4}[-.\s()]?\d{1,9}'
    # Filter out very short numbers that might be false positives
    return [match for match in re.findall(phone_regex, text) if len(re.sub(r'\D', '', match)) >= 7]

def extract_locations(text: str) -> List[str]:
    """Extracts locations (GPE entities) from text using spaCy's NER."""
    doc = nlp(text)
    locations = [ent.text.strip() for ent in doc.ents if ent.label_ == "GPE"]
    return list(set(locations)) # Return unique locations

def extract_skills(text: str, skill_keywords: List[str] = None) -> List[str]:
    """
    Extracts skills from text.
    Can use a predefined list of keywords or attempt generic extraction.
    For now, uses a simple keyword matching approach after text cleaning.
    """
    if skill_keywords is None:
        # Placeholder for more advanced skill extraction if no keywords are provided.
        # For now, this will be empty if no skill_keywords are passed.
        # A more robust version might use part-of-speech tagging or a pre-trained skill model.
        skill_keywords = [
            "python", "java", "c++", "javascript", "react", "angular", "vue",
            "node.js", "django", "flask", "fastapi", "spring",
            "sql", "nosql", "mongodb", "postgresql", "mysql",
            "docker", "kubernetes", "aws", "azure", "gcp",
            "machine learning", "deep learning", "nlp", "computer vision",
            "data analysis", "data science", "pandas", "numpy", "scikit-learn",
            "tensorflow", "pytorch", "spacy", "transformers"
        ]

    found_skills = []
    # Use word boundaries for more precise matching of keywords
    for skill in skill_keywords:
        # Escape special characters in skill for regex, e.g. C++
        escaped_skill = re.escape(skill)
        if re.search(r'\b' + escaped_skill + r'\b', text, re.IGNORECASE):
            found_skills.append(skill)
    return list(set(found_skills))

def extract_projects_and_internships(text: str) -> Dict[str, List[str]]:
    """
    Placeholder for extracting projects and internships.
    This would typically involve section detection and then parsing content within those sections.
    A simple keyword-based approach is used here as a starting point.
    """
    # This is highly heuristic and would need significant improvement for real-world use.
    projects = []
    internships = []

    # Try to find sections; assumes simple headers. Case-insensitive search.
    project_section = re.search(r"(projects|personal projects|portfolio)(.+?)(experience|internships|education|skills|awards|$)", text, re.IGNORECASE | re.DOTALL)
    if project_section:
        # Further parse project_section.group(2) for individual projects
        # For now, just taking the whole block.
        projects.append(project_section.group(2).strip())

    internship_section = re.search(r"(internships|work experience|experience)(.+?)(projects|education|skills|awards|$)", text, re.IGNORECASE | re.DOTALL)
    if internship_section:
        # Further parse internship_section.group(2) for individual internships
        internships.append(internship_section.group(2).strip())

    # If no distinct sections are found, could try sentence-level keyword spotting (less accurate)
    if not projects:
        for sentence in text.split('.'): # Rough sentence split
            if "project" in sentence.lower() and len(sentence.split()) > 5: # Avoid trivial mentions
                 projects.append(sentence.strip())

    if not internships:
        for sentence in text.split('.'):
            if ("intern" in sentence.lower() or "internship" in sentence.lower()) and len(sentence.split()) > 5:
                 internships.append(sentence.strip())

    return {"projects": list(set(projects)), "internships": list(set(internships))}


def parse_resume_text(resume_text: str) -> Dict[str, Any]:
    """
    Parses the raw text of a resume to extract structured information.
    """
    cleaned_original_text = resume_text # Keep original for some regex if needed before cleaning
    cleaned_text_for_nlp = clean_text(resume_text) # Cleaned for NLP and keyword search

    name = extract_name(cleaned_text_for_nlp) # Try on cleaned text first
    if not name: # If name not found in cleaned, try original (e.g. if cleaning removed vital parts)
        # Create a doc from the first few lines of the original text, often where names are
        doc_orig = nlp(' '.join(cleaned_original_text.split('\n')[:5]))
        for ent in doc_orig.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                break

    emails = extract_emails(resume_text) # Emails are better extracted from original text due to special chars
    phones = extract_phones(resume_text)
    locations = extract_locations(cleaned_text_for_nlp)
    skills = extract_skills(cleaned_text_for_nlp) # Uses the default list of skills

    # Projects and internships extraction is very basic for now
    sections = extract_projects_and_internships(cleaned_text_for_nlp)

    return {
        "name": name,
        "email": emails[0] if emails else None, # Take the first email
        "phone": phones[0] if phones else None, # Take the first phone
        "locations": locations,
        "skills": skills,
        "projects": sections["projects"],
        "internships": sections["internships"],
        "cleaned_text": cleaned_text_for_nlp # Include cleaned text for embedding later
    }

if __name__ == '__main__':
    print("Running resume_parser.py directly for testing...")
    sample_resume_text = """
    John Michael Doe
    john.doe@example.com | (123) 456-7890 | New York, NY

    Summary
    Highly skilled software engineer with 5 years of experience in Python and Java.
    Based in San Francisco but open to roles in Chicago.

    Experience
    Software Engineer at Tech Solutions Inc. (Jan 2020 - Present)
    - Developed web applications using Django and React.
    - Led a team of 3 developers.

    Internship at Web Innovations LLC (Summer 2019)
    - Worked on a project involving data analysis with Pandas.
    - My main project was a recommendation engine.

    Projects
    Personal Portfolio Website
    - Built using Flask and deployed on AWS.
    My Cool Project
    - A machine learning model to predict stock prices.

    Education
    M.S. in Computer Science, Stanford University
    B.S. in Software Engineering, MIT

    Skills
    Python, Java, C++, JavaScript, React, Django, Flask, Pandas, NumPy, SQL, Docker, AWS, Machine Learning
    """

    parsed_data = parse_resume_text(sample_resume_text)
    print("\n--- Parsed Resume Data ---")
    for key, value in parsed_data.items():
        if key == "cleaned_text":
            # print(f"{key.capitalize()}: {value[:100]}...") # Avoid printing full cleaned text
            continue
        print(f"{key.capitalize()}: {value}")

    # Test with only a name and email
    sample_short_resume = "Jane Doe jane.d@test.co Some City, CA"
    parsed_short_data = parse_resume_text(sample_short_resume)
    print("\n--- Parsed Short Resume Data ---")
    for key, value in parsed_short_data.items():
        if key == "cleaned_text":
            continue
        print(f"{key.capitalize()}: {value}")

    # Test with a different structure
    sample_resume_text_2 = """
    Dr. Emily White, PhD
    emily.white@email.lab | London, UK | +44 20 7946 0958

    PROFILE
    A dedicated researcher with expertise in Natural Language Processing and Deep Learning.
    Key skills include: Python, TensorFlow, PyTorch, spaCy, and Transformers.

    PROJECT WORK
    Sentiment Analysis Tool (NLP)
    - Developed a tool for real-time sentiment classification.

    PROFESSIONAL EXPERIENCE
    Research Intern at AI Research Ltd. (June 2021 - Aug 2021)
    - Contributed to a cutting-edge project on semantic search.
    """
    parsed_data_2 = parse_resume_text(sample_resume_text_2)
    print("\n--- Parsed Resume Data 2 ---")
    for key, value in parsed_data_2.items():
        if key == "cleaned_text":
            continue
        print(f"{key.capitalize()}: {value}")
