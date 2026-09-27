import streamlit as st
import fitz # PyMuPDF
import pandas as pd
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- PAGE SETUP (FRONTEND) ---
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")
st.title("📄 AI Resume Analyzer & Job Matcher")
st.markdown("### AIML Mini Project - ATS Score Checker")

# --- BACKEND: Load AI Model ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_skills():
    try:
        with open("skills.csv", "r") as f:
            content = f.read().lower()
            skills = [s.strip() for s in re.split(r'[,\n]', content) if s.strip()]
        return list(set(skills))
    except:
        return ["python","java","sql","machine learning","nlp","tensorflow","aws","docker","react","html","css","javascript"]

model = load_model()
ALL_SKILLS = load_skills()

# --- BACKEND: Functions ---
def extract_text_from_pdf(pdf_file):
    text = ""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

def extract_skills(text):
    text_lower = text.lower()
    found = []
    for skill in ALL_SKILLS:
        # Use word boundary for better matching
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
            found.append(skill)
    return list(set(found))

def get_match_score(resume_text, jd_text):
    embeddings = model.encode([resume_text, jd_text])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(score * 100, 2)

# --- FRONTEND: UI ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Your Resume (PDF)")
    resume_file = st.file_uploader("Choose PDF file", type=["pdf"], key="resume")
    if resume_file:
        st.success("Resume Uploaded!")

with col2:
    st.subheader("2. Paste Job Description")
    jd_text = st.text_area("Paste JD here", height=250, placeholder="Ex: Looking for Python Developer with skills: Python, ML, SQL, AWS, Docker...")

# --- FRONTEND + BACKEND CONNECTION ---
if st.button("🚀 Analyze Resume", type="primary"):
    if not resume_file or not jd_text.strip():
        st.error("⚠️ Please upload resume AND paste Job Description both!")
    else:
        with st.spinner("AI is analyzing..."):
            resume_text = extract_text_from_pdf(resume_file)
            score = get_match_score(resume_text, jd_text)
            resume_skills = extract_skills(resume_text)
            jd_skills = extract_skills(jd_text)
            missing_skills = [s for s in jd_skills if s not in resume_skills]

        st.divider()
        st.subheader("📊 Results")

        m1, m2, m3 = st.columns(3)
        m1.metric("ATS Match Score", f"{score}%")
        m2.metric("Skills Found", len(resume_skills))
        m3.metric("Missing Skills", len(missing_skills))

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            title = {'text': "Match Score"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "green" if score > 70 else "orange" if score > 50 else "red"},
                'steps': [
                    {'range': [0, 50], 'color': "#ffcccc"},
                    {'range': [50, 75], 'color': "#fff4cc"},
                    {'range': [75, 100], 'color': "#ccffcc"}]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("✅ Your Skills")
            if resume_skills:
                for s in resume_skills:
                    st.markdown(f"- {s}")
            else:
                st.write("No skills detected")

        with c2:
            st.subheader("❌ Missing Skills (Add these)")
            if missing_skills:
                for ms in missing_skills:
                    st.markdown(f"- **{ms}**")
            else:
                st.success("Perfect! No missing skills")

        st.subheader("💡 Suggestion for Viva")
        if score < 50:
            st.warning("LOW Match. Add more JD keywords in your Skills and Projects section.")
        elif score < 75:
            st.info("GOOD Match. Add the missing skills in your resume to get 85%+ score.")
        else:
            st.success("EXCELLENT! Your resume is ready to apply. ATS will select it.")

        with st.expander("See Full Resume Text"):
            st.text(resume_text[:3000])