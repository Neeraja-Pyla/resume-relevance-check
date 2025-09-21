import streamlit as st
import re, string
import PyPDF2
#from pypdf import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# -------------------------------
# Helpers
# -------------------------------
def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_name(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        return lines[0]
    return "Unknown Candidate"

def give_feedback(score):
    if score >= 75:
        return "High", "Strong alignment with JD. Highlight manufacturing/data projects in applications."
    elif score >= 50:
        return "Medium", "Good foundation. Improve manufacturing domain knowledge, Excel/Python projects, and practical internships."
    else:
        return "Low", "Focus on building core skills: Python (Pandas), SQL, manufacturing data handling. Add internships or projects closer to JD."

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📊 Resume Relevance Checker")

jd_file = st.file_uploader("Upload Job Description (PDF)", type="pdf")
resume_files = st.file_uploader("Upload Resumes (PDFs)", type="pdf", accept_multiple_files=True)

if jd_file and resume_files:
    # Extract JD text
    jd_text = clean_text(extract_text_from_pdf(jd_file))
    
    # Extract resumes
    resumes = {}
    names = {}
    for pdf in resume_files:
        raw_text = extract_text_from_pdf(pdf)
        resumes[pdf.name] = clean_text(raw_text)
        names[pdf.name] = extract_name(raw_text)

    # TF-IDF similarity
    documents = [jd_text] + list(resumes.values())
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    scores = (similarities * 100).round(2)

    # Ranking
    ranking = sorted(zip(resumes.keys(), scores), key=lambda x: x[1], reverse=True)
    results = []
    for i, (fname, score) in enumerate(ranking, 1):
        suitability, feedback = give_feedback(score)
        results.append({
            "Rank": i,
            "Name": names[fname],
            "File": fname,
            "Relevance Score (%)": score,
            "Suitability": suitability,
            "Feedback": feedback
        })
    
    df = pd.DataFrame(results)
    st.subheader("📋 Results")
    st.dataframe(df)
