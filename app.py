import streamlit as st
import re
import io
import pandas as pd
import nltk
import spacy

from pypdf import PdfReader
from docx import Document

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional advanced semantic NLP
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM UI
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 17px;
    opacity: 0.95;
}

.card {
    padding: 22px;
    border-radius: 18px;
    background: white;
    border: 1px solid #e7eaf0;
    box-shadow: 0 5px 20px rgba(0,0,0,0.06);
    margin-bottom: 18px;
}

.metric-card {
    padding: 20px;
    border-radius: 16px;
    background: white;
    text-align: center;
    border: 1px solid #e7eaf0;
    box-shadow: 0 5px 18px rgba(0,0,0,0.05);
}

.metric-number {
    font-size: 32px;
    font-weight: 700;
}

.metric-title {
    font-size: 14px;
    color: #666;
}

.skill {
    display: inline-block;
    padding: 7px 12px;
    margin: 4px;
    border-radius: 20px;
    background: #eef2ff;
    color: #4f46e5;
    font-size: 13px;
    font-weight: 600;
}

.missing {
    display: inline-block;
    padding: 7px 12px;
    margin: 4px;
    border-radius: 20px;
    background: #fff1f2;
    color: #e11d48;
    font-size: 13px;
    font-weight: 600;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# NLTK SETUP
# =========================================================

@st.cache_resource
def setup_nltk():

    packages = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet")
    ]

    for resource, package in packages:

        try:
            nltk.data.find(resource)

        except LookupError:
            nltk.download(package, quiet=True)


setup_nltk()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# =========================================================
# SPACY
# =========================================================

@st.cache_resource
def load_spacy():

    try:
        return spacy.load("en_core_web_sm")

    except Exception:
        return None


nlp = load_spacy()


# =========================================================
# SENTENCE TRANSFORMER
# =========================================================

@st.cache_resource
def load_sentence_model():

    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None

    try:

        return SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    except Exception:

        return None


semantic_model = load_sentence_model()


# =========================================================
# SKILL DATABASE
# =========================================================

SKILL_ALIASES = {

    "python": ["python"],
    "java": ["java"],
    "c": ["c programming"],
    "c++": ["c++"],
    "c#": ["c#"],

    "javascript": [
        "javascript",
        "js"
    ],

    "typescript": [
        "typescript",
        "ts"
    ],

    "html": [
        "html",
        "html5"
    ],

    "css": [
        "css",
        "css3"
    ],

    "react": [
        "react",
        "react.js",
        "reactjs"
    ],

    "node.js": [
        "node.js",
        "nodejs",
        "node"
    ],

    "django": [
        "django"
    ],

    "flask": [
        "flask"
    ],

    "fastapi": [
        "fastapi"
    ],

    "sql": [
        "sql"
    ],

    "mysql": [
        "mysql"
    ],

    "postgresql": [
        "postgresql",
        "postgres"
    ],

    "mongodb": [
        "mongodb",
        "mongo"
    ],

    "machine learning": [
        "machine learning",
        "ml"
    ],

    "deep learning": [
        "deep learning",
        "dl"
    ],

    "natural language processing": [
        "natural language processing",
        "nlp"
    ],

    "computer vision": [
        "computer vision",
        "cv"
    ],

    "artificial intelligence": [
        "artificial intelligence",
        "ai"
    ],

    "tensorflow": [
        "tensorflow"
    ],

    "pytorch": [
        "pytorch"
    ],

    "keras": [
        "keras"
    ],

    "scikit-learn": [
        "scikit-learn",
        "sklearn"
    ],

    "pandas": [
        "pandas"
    ],

    "numpy": [
        "numpy"
    ],

    "matplotlib": [
        "matplotlib"
    ],

    "seaborn": [
        "seaborn"
    ],

    "transformers": [
        "transformers"
    ],

    "bert": [
        "bert"
    ],

    "llm": [
        "llm",
        "large language model"
    ],

    "generative ai": [
        "generative ai",
        "genai"
    ],

    "data science": [
        "data science"
    ],

    "data analysis": [
        "data analysis",
        "data analytics"
    ],

    "power bi": [
        "power bi"
    ],

    "tableau": [
        "tableau"
    ],

    "git": [
        "git"
    ],

    "github": [
        "github"
    ],

    "docker": [
        "docker"
    ],

    "aws": [
        "aws",
        "amazon web services"
    ],

    "azure": [
        "azure",
        "microsoft azure"
    ],

    "linux": [
        "linux"
    ],

    "rest api": [
        "rest api",
        "restful api"
    ],

    "firebase": [
        "firebase"
    ],

    "communication": [
        "communication",
        "communication skills"
    ],

    "leadership": [
        "leadership",
        "leadership skills"
    ],

    "teamwork": [
        "teamwork",
        "team player"
    ],

    "problem solving": [
        "problem solving",
        "problem-solving"
    ]
}


# =========================================================
# FILE EXTRACTION
# =========================================================

def extract_pdf(file):

    reader = PdfReader(file)

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def extract_docx(file):

    data = file.read()

    document = Document(
        io.BytesIO(data)
    )

    paragraphs = []

    for paragraph in document.paragraphs:

        if paragraph.text.strip():

            paragraphs.append(
                paragraph.text
            )

    return "\n".join(paragraphs)


def extract_text(file):

    filename = file.name.lower()

    if filename.endswith(".pdf"):
        return extract_pdf(file)

    elif filename.endswith(".docx"):
        return extract_docx(file)

    elif filename.endswith(".txt"):

        return file.read().decode(
            "utf-8",
            errors="ignore"
        )

    return ""


# =========================================================
# NLP PREPROCESSING
# =========================================================

def preprocess(text):

    text = text.lower()

    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z0-9+#.\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    tokens = re.findall(
        r"[a-zA-Z0-9+#.]+",
        text
    )

    result = []

    for token in tokens:

        token = token.strip(".")

        if (
            token
            and token not in stop_words
            and len(token) > 1
        ):

            result.append(
                lemmatizer.lemmatize(token)
            )

    return " ".join(result)


# =========================================================
# EMAIL
# =========================================================

def extract_email(text):

    match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    return (
        match.group(0)
        if match
        else "Not found"
    )


# =========================================================
# PHONE
# =========================================================

def extract_phone(text):

    patterns = [

        r"\b01[3-9]\d{8}\b",

        r"\+?\d[\d\s().-]{8,}\d"

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text
        )

        if match:

            return match.group(0).strip()

    return "Not found"


# =========================================================
# SKILL EXTRACTION
# =========================================================

def extract_skills(text):

    lower_text = text.lower()

    found = []

    for canonical, aliases in SKILL_ALIASES.items():

        for alias in aliases:

            pattern = (
                r"(?<![a-z0-9+#])"
                + re.escape(alias)
                + r"(?![a-z0-9+#])"
            )

            if re.search(
                pattern,
                lower_text
            ):

                found.append(canonical)

                break

    return sorted(set(found))


# =========================================================
# NER
# =========================================================

def extract_entities(text):

    if nlp is None:

        return []

    try:

        doc = nlp(
            text[:100000]
        )

        data = []

        for ent in doc.ents:

            if ent.label_ in [
                "PERSON",
                "ORG",
                "GPE",
                "DATE",
                "CARDINAL",
                "FAC",
                "PRODUCT"
            ]:

                data.append({

                    "Entity": ent.text,

                    "Type": ent.label_

                })

        return data

    except Exception:

        return []


# =========================================================
# RESUME SECTION DETECTION
# =========================================================

def detect_sections(text):

    sections = {

        "Education": [],
        "Experience": [],
        "Skills": [],
        "Projects": [],
        "Certifications": [],
        "Achievements": []

    }

    lines = text.splitlines()

    current_section = None

    section_patterns = {

        "Education": [
            "education",
            "academic background",
            "qualification"
        ],

        "Experience": [
            "experience",
            "work experience",
            "employment"
        ],

        "Skills": [
            "skills",
            "technical skills",
            "core skills"
        ],

        "Projects": [
            "projects",
            "project experience"
        ],

        "Certifications": [
            "certifications",
            "certificates"
        ],

        "Achievements": [
            "achievements",
            "awards"
        ]

    }

    for line in lines:

        clean = line.strip().lower()

        found_section = False

        for section, patterns in section_patterns.items():

            for pattern in patterns:

                if clean == pattern:

                    current_section = section

                    found_section = True

                    break

            if found_section:
                break

        if found_section:
            continue

        if current_section:

            sections[current_section].append(
                line
            )

    return {
        key: "\n".join(value).strip()
        for key, value in sections.items()
    }


# =========================================================
# EDUCATION EXTRACTION
# =========================================================

def extract_education(text):

    education_keywords = [

        "b.sc",
        "bsc",
        "bachelor",
        "b.tech",
        "m.sc",
        "msc",
        "master",
        "m.tech",
        "computer science",
        "computer engineering",
        "software engineering",
        "information technology"

    ]

    lines = []

    for line in text.splitlines():

        lower = line.lower()

        for keyword in education_keywords:

            if keyword in lower:

                lines.append(
                    line.strip()
                )

                break

    return list(
        dict.fromkeys(lines)
    )[:10]


# =========================================================
# EXPERIENCE EXTRACTION
# =========================================================

def extract_experience(text):

    patterns = [

        r"\b\d+\+?\s+years?\b",

        r"\b\d+\+?\s+months?\b",

        r"\b\d+\s*-\s*\d+\s+years?\b"

    ]

    results = []

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text.lower()
        )

        results.extend(matches)

    return list(
        dict.fromkeys(results)
    )


# =========================================================
# KEYWORD EXTRACTION
# =========================================================

def extract_keywords(text, top_n=15):

    processed = preprocess(text)

    if not processed:

        return []

    try:

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=100
        )

        matrix = vectorizer.fit_transform(
            [processed]
        )

        scores = matrix.toarray()[0]

        words = vectorizer.get_feature_names_out()

        ranked = sorted(
            zip(words, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            word
            for word, score in ranked[:top_n]
        ]

    except Exception:

        return []


# =========================================================
# TF-IDF SIMILARITY
# =========================================================

def tfidf_similarity(
    resume,
    job
):

    resume_text = preprocess(
        resume
    )

    job_text = preprocess(
        job
    )

    if not resume_text or not job_text:

        return 0.0

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000
    )

    matrix = vectorizer.fit_transform(

        [
            resume_text,
            job_text
        ]

    )

    score = cosine_similarity(

        matrix[0:1],
        matrix[1:2]

    )[0][0]

    return round(
        score * 100,
        2
    )


# =========================================================
# SEMANTIC SIMILARITY
# =========================================================

def semantic_similarity(
    resume,
    job
):

    if semantic_model is None:

        return None

    try:

        resume_embedding = semantic_model.encode(
            resume[:15000],
            normalize_embeddings=True
        )

        job_embedding = semantic_model.encode(
            job[:15000],
            normalize_embeddings=True
        )

        score = float(
            resume_embedding @ job_embedding
        )

        score = max(
            0,
            min(
                100,
                score * 100
            )
        )

        return round(
            score,
            2
        )

    except Exception:

        return None


# =========================================================
# SKILL SCORE
# =========================================================

def skill_match_score(
    resume_skills,
    job_skills
):

    if not job_skills:

        return 0.0

    matched = (
        set(resume_skills)
        &
        set(job_skills)
    )

    return round(

        len(matched)
        /
        len(set(job_skills))
        *
        100,

        2

    )


# =========================================================
# FINAL SCORE
# =========================================================

def final_score(
    tfidf,
    semantic,
    skill
):

    if semantic is not None:

        score = (

            tfidf * 0.25

            +

            semantic * 0.50

            +

            skill * 0.25

        )

    else:

        score = (

            tfidf * 0.70

            +

            skill * 0.30

        )

    return round(
        score,
        2
    )


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

<h1>🤖 AI Resume Analyzer</h1>

<p>
NLP-Powered Resume Parsing & Intelligent Job Matching System
</p>

<p>
Extract • Understand • Compare • Match
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🧠 NLP Engine")

    st.write(
        "This application uses:"
    )

    st.write(
        """
        ✅ Text preprocessing

        ✅ Stopword removal

        ✅ Lemmatization

        ✅ Skill normalization

        ✅ Named Entity Recognition

        ✅ Section detection

        ✅ TF-IDF

        ✅ Cosine Similarity

        ✅ Semantic Similarity

        ✅ Keyword extraction
        """
    )

    st.markdown("---")

    if semantic_model:

        st.success(
            "Semantic NLP: Active"
        )

    else:

        st.warning(
            "Semantic NLP unavailable. "
            "TF-IDF matching will still work."
        )


# =========================================================
# JOB DESCRIPTION
# =========================================================

st.markdown(
    '<div class="section-title">💼 Job Description</div>',
    unsafe_allow_html=True
)

job_description = st.text_area(

    "Paste the target Job Description:",

    height=230,

    placeholder="""
Example:

We are looking for an NLP Engineer with experience
in Python, Machine Learning, Natural Language Processing,
Transformers, SQL and Deep Learning.
The candidate should have strong communication
and teamwork skills.
"""

)


job_skills = []

if job_description:

    job_skills = extract_skills(
        job_description
    )

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.write(
        "### 🎯 Required Skills"
    )

    if job_skills:

        skills_html = ""

        for skill in job_skills:

            skills_html += (
                f'<span class="skill">{skill}</span>'
            )

        st.markdown(
            skills_html,
            unsafe_allow_html=True
        )

    else:

        st.info(
            "No predefined skills detected."
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# FILE UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">📄 Upload Resumes</div>',
    unsafe_allow_html=True
)

uploaded_files = st.file_uploader(

    "Upload one or multiple resumes",

    type=[
        "pdf",
        "docx",
        "txt"
    ],

    accept_multiple_files=True

)


# =========================================================
# ANALYSIS
# =========================================================

if uploaded_files and job_description:

    st.markdown(
        '<div class="section-title">🔍 AI Analysis</div>',
        unsafe_allow_html=True
    )

    results = []

    details = {}


    progress = st.progress(0)

    total = len(
        uploaded_files
    )


    for index, file in enumerate(
        uploaded_files
    ):

        try:

            raw_text = extract_text(
                file
            )

            if not raw_text.strip():

                continue


            processed = preprocess(
                raw_text
            )


            resume_skills = extract_skills(
                raw_text
            )


            tfidf_score = tfidf_similarity(

                raw_text,

                job_description

            )


            semantic_score = semantic_similarity(

                raw_text,

                job_description

            )


            skill_score = skill_match_score(

                resume_skills,

                job_skills

            )


            overall = final_score(

                tfidf_score,

                semantic_score,

                skill_score

            )


            matched_skills = sorted(

                set(resume_skills)
                &
                set(job_skills)

            )


            missing_skills = sorted(

                set(job_skills)
                -
                set(resume_skills)

            )


            entities = extract_entities(
                raw_text
            )


            sections = detect_sections(
                raw_text
            )


            education = extract_education(
                raw_text
            )


            experience = extract_experience(
                raw_text
            )


            keywords = extract_keywords(
                raw_text
            )


            results.append({

                "Resume":
                    file.name,

                "Overall Match %":
                    overall,

                "Semantic %":
                    semantic_score
                    if semantic_score is not None
                    else "N/A",

                "TF-IDF %":
                    tfidf_score,

                "Skill Match %":
                    skill_score,

                "Matched Skills":
                    len(matched_skills),

                "Missing Skills":
                    len(missing_skills)

            })


            details[file.name] = {

                "raw":
                    raw_text,

                "processed":
                    processed,

                "skills":
                    resume_skills,

                "matched":
                    matched_skills,

                "missing":
                    missing_skills,

                "entities":
                    entities,

                "sections":
                    sections,

                "education":
                    education,

                "experience":
                    experience,

                "keywords":
                    keywords,

                "email":
                    extract_email(
                        raw_text
                    ),

                "phone":
                    extract_phone(
                        raw_text
                    )

            }


        except Exception as error:

            st.error(
                f"Error processing "
                f"{file.name}: {error}"
            )


        progress.progress(
            (index + 1) / total
        )


    # =====================================================
    # RESULTS
    # =====================================================

    if results:

        df = pd.DataFrame(
            results
        )

        df = df.sort_values(

            "Overall Match %",

            ascending=False

        ).reset_index(
            drop=True
        )


        st.success(
            f"Successfully analyzed "
            f"{len(df)} resume(s)."
        )


        # =================================================
        # TOP METRICS
        # =================================================

        c1, c2, c3, c4 = st.columns(4)


        c1.metric(
            "📄 Resumes",
            len(df)
        )


        c2.metric(
            "🎯 Job Skills",
            len(job_skills)
        )


        c3.metric(
            "🧠 NLP Engine",
            "Active"
            if semantic_model
            else "TF-IDF"
        )


        c4.metric(
            "🔎 NER",
            "Active"
            if nlp
            else "Unavailable"
        )


        # =================================================
        # RESULTS TABLE
        # =================================================

        st.markdown(
            '<div class="section-title">📊 Candidate Comparison</div>',
            unsafe_allow_html=True
        )


        st.dataframe(

            df,

            use_container_width=True,

            hide_index=True

        )


        # =================================================
        # CSV
        # =================================================

        csv = df.to_csv(
            index=False
        ).encode(
            "utf-8"
        )


        st.download_button(

            "⬇️ Download Candidate Results",

            csv,

            "resume_analysis_results.csv",

            "text/csv"

        )


        # =================================================
        # CANDIDATE DETAILS
        # =================================================

        st.markdown(
            '<div class="section-title">👤 Candidate Analysis</div>',
            unsafe_allow_html=True
        )


        for _, row in df.iterrows():

            filename = row[
                "Resume"
            ]

            data = details[
                filename
            ]


            with st.expander(

                f"📄 {filename}  |  "
                f"Match: {row['Overall Match %']}%"

            ):


                # -----------------------------------------
                # SCORE CARDS
                # -----------------------------------------

                a, b, c, d = st.columns(4)


                a.metric(
                    "🎯 Overall",
                    f"{row['Overall Match %']}%"
                )


                b.metric(
                    "🧠 Semantic",
                    (
                        f"{row['Semantic %']}%"
                        if row["Semantic %"] != "N/A"
                        else "N/A"
                    )
                )


                c.metric(
                    "📚 TF-IDF",
                    f"{row['TF-IDF %']}%"
                )


                d.metric(
                    "🛠 Skills",
                    f"{row['Skill Match %']}%"
                )


                # -----------------------------------------
                # CONTACT
                # -----------------------------------------

                st.write(
                    "### 📧 Contact Information"
                )

                st.write(
                    f"**Email:** {data['email']}"
                )

                st.write(
                    f"**Phone:** {data['phone']}"
                )


                # -----------------------------------------
                # SKILLS
                # -----------------------------------------

                st.write(
                    "### 🧠 Detected Skills"
                )


                skill_html = ""

                for skill in data["skills"]:

                    skill_html += (
                        f'<span class="skill">{skill}</span>'
                    )


                if skill_html:

                    st.markdown(
                        skill_html,
                        unsafe_allow_html=True
                    )

                else:

                    st.write(
                        "No skills detected."
                    )


                # -----------------------------------------
                # MATCHED SKILLS
                # -----------------------------------------

                st.write(
                    "### ✅ Matched Skills"
                )


                matched_html = ""

                for skill in data["matched"]:

                    matched_html += (
                        f'<span class="skill">{skill}</span>'
                    )


                if matched_html:

                    st.markdown(
                        matched_html,
                        unsafe_allow_html=True
                    )

                else:

                    st.write(
                        "No matched skills."
                    )


                # -----------------------------------------
                # MISSING SKILLS
                # -----------------------------------------

                st.write(
                    "### ⚠️ Missing Skills"
                )


                missing_html = ""

                for skill in data["missing"]:

                    missing_html += (
                        f'<span class="missing">{skill}</span>'
                    )


                if missing_html:

                    st.markdown(
                        missing_html,
                        unsafe_allow_html=True
                    )

                else:

                    st.success(
                        "No predefined job skills are missing."
                    )


                # -----------------------------------------
                # EDUCATION
                # -----------------------------------------

                st.write(
                    "### 🎓 Education"
                )


                if data["education"]:

                    for item in data["education"]:

                        st.write(
                            f"• {item}"
                        )

                else:

                    st.write(
                        "Education information not clearly detected."
                    )


                # -----------------------------------------
                # EXPERIENCE
                # -----------------------------------------

                st.write(
                    "### 💼 Experience Indicators"
                )


                if data["experience"]:

                    st.write(
                        ", ".join(
                            data["experience"]
                        )
                    )

                else:

                    st.write(
                        "No explicit experience duration detected."
                    )


                # -----------------------------------------
                # KEYWORDS
                # -----------------------------------------

                st.write(
                    "### 🔑 Important NLP Keywords"
                )


                if data["keywords"]:

                    st.write(
                        " • ".join(
                            data["keywords"]
                        )
                    )


                # -----------------------------------------
                # NER
                # -----------------------------------------

                st.write(
                    "### 🔎 Named Entities"
                )


                if data["entities"]:

                    entity_df = pd.DataFrame(
                        data["entities"]
                    )

                    st.dataframe(
                        entity_df,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.write(
                        "No named entities detected."
                    )


                # -----------------------------------------
                # RESUME SECTIONS
                # -----------------------------------------

                st.write(
                    "### 📑 Resume Sections"
                )


                for section, content in data[
                    "sections"
                ].items():

                    if content:

                        with st.expander(
                            section
                        ):

                            st.write(
                                content[:5000]
                            )


                # -----------------------------------------
                # PROCESSED TEXT
                # -----------------------------------------

                st.write(
                    "### 📝 NLP Processed Text"
                )


                st.text_area(

                    "Cleaned + Tokenized + Lemmatized",

                    data["processed"][:5000],

                    height=180,

                    key=
                    "processed_"
                    + filename

                )


# =========================================================
# NO JOB DESCRIPTION
# =========================================================

elif uploaded_files and not job_description:

    st.warning(
        "⚠️ Please paste a Job Description first."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "AI Resume Analyzer | "
    "Python • NLTK • spaCy • "
    "TF-IDF • Sentence Transformers • "
    "Cosine Similarity • Streamlit"
)
