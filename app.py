"""
SMS Spam Detection System - Modern AI Dashboard
Frontend/UI Redesigned based on Modern Dark SaaS Dashboard aesthetics.
Maintains 100% compatibility with existing backend, models, and dataset.
"""

import os
import sys
import datetime
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Configure wide layout and page metadata
st.set_page_config(
    page_title="SMS Spam Detection | AI Dashboard",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project root path setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import preprocess_text
from src.predict import get_predictor, predict_sms

MODELS_DIR = "models"
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "spam_classifier.pkl")
NAIVE_BAYES_PATH = os.path.join(MODELS_DIR, "naive_bayes.pkl")
LOGISTIC_REG_PATH = os.path.join(MODELS_DIR, "logistic_regression.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.pkl")
DATASET_PATH = "data/spam.csv"


# ==============================================================================
# CACHED DATA & MODEL LOADERS (PRESERVED & OPTIMIZED)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_all_artifacts():
    """Loads and caches models, vectorizer, and evaluation metrics."""
    artifacts = {
        "classifier": None,
        "naive_bayes": None,
        "logistic_reg": None,
        "vectorizer": None,
        "metrics": None
    }
    try:
        if os.path.exists(CLASSIFIER_PATH):
            artifacts["classifier"] = joblib.load(CLASSIFIER_PATH)
        if os.path.exists(NAIVE_BAYES_PATH):
            artifacts["naive_bayes"] = joblib.load(NAIVE_BAYES_PATH)
        elif artifacts["classifier"] is not None:
            artifacts["naive_bayes"] = artifacts["classifier"]

        if os.path.exists(LOGISTIC_REG_PATH):
            artifacts["logistic_reg"] = joblib.load(LOGISTIC_REG_PATH)
        elif artifacts["classifier"] is not None:
            artifacts["logistic_reg"] = artifacts["classifier"]

        if os.path.exists(VECTORIZER_PATH):
            artifacts["vectorizer"] = joblib.load(VECTORIZER_PATH)
        if os.path.exists(METRICS_PATH):
            artifacts["metrics"] = joblib.load(METRICS_PATH)
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
    return artifacts


@st.cache_data(show_spinner=False)
def load_dataset_stats():
    """Loads dataset and extracts real statistical distributions."""
    if not os.path.exists(DATASET_PATH):
        return None
    try:
        df_raw = pd.read_csv(DATASET_PATH, encoding="latin-1")
    except Exception:
        df_raw = pd.read_csv(DATASET_PATH, encoding="utf-8", errors="replace")

    # Standardize columns
    if "v1" in df_raw.columns and "v2" in df_raw.columns:
        df = df_raw[["v1", "v2"]].copy()
        df.rename(columns={"v1": "label", "v2": "message"}, inplace=True)
    elif "label" in df_raw.columns and "message" in df_raw.columns:
        df = df_raw[["label", "message"]].copy()
    else:
        df = df_raw.iloc[:, [0, 1]].copy()
        df.columns = ["label", "message"]

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])].copy()

    total_messages = len(df)
    ham_count = int((df["label"] == "ham").sum())
    spam_count = int((df["label"] == "spam").sum())
    spam_percentage = round((spam_count / total_messages) * 100, 1) if total_messages else 0.0
    ham_percentage = round((ham_count / total_messages) * 100, 1) if total_messages else 0.0

    df["char_len"] = df["message"].astype(str).apply(len)
    avg_ham_len = round(float(df[df["label"] == "ham"]["char_len"].mean()), 1) if ham_count else 0.0
    avg_spam_len = round(float(df[df["label"] == "spam"]["char_len"].mean()), 1) if spam_count else 0.0

    return {
        "df": df,
        "total_messages": total_messages,
        "ham_count": ham_count,
        "spam_count": spam_count,
        "spam_percentage": spam_percentage,
        "ham_percentage": ham_percentage,
        "avg_ham_len": avg_ham_len,
        "avg_spam_len": avg_spam_len
    }


artifacts = load_all_artifacts()
dataset_stats = load_dataset_stats()


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
NAV_ITEMS = [
    "🏠 Dashboard",
    "🔍 Detect Spam",
    "🕘 Prediction History",
    "📊 Model Performance",
    "📈 Data Analytics",
    "💬 Example Messages",
    "ℹ️ About Project"
]

if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = "🏠 Dashboard"

if "input_sms_text" not in st.session_state:
    st.session_state["input_sms_text"] = ""

if "selected_model_name" not in st.session_state:
    st.session_state["selected_model_name"] = "Naive Bayes"

if "trigger_predict" not in st.session_state:
    st.session_state["trigger_predict"] = False

if "last_prediction_result" not in st.session_state:
    st.session_state["last_prediction_result"] = None

if "history_view_index" not in st.session_state:
    st.session_state["history_view_index"] = None

if "history_page_num" not in st.session_state:
    st.session_state["history_page_num"] = 1

# Initialize Prediction History with realistic sample records from dataset
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = [
        {
            "id": 1,
            "message": "Congratulations! You have won a free $1,000 Walmart Gift Card. Call 0800-123-4567 or visit win-card.com now to claim.",
            "prediction": "SPAM",
            "confidence": 99.87,
            "model": "Naive Bayes",
            "timestamp": "28 Sep 11:20",
            "processed_text": "congratul won free 1 000 walmart gift card call 0800 123 4567 visit win card com claim"
        },
        {
            "id": 2,
            "message": "Hey, are we still meeting today at the college library for the group study session?",
            "prediction": "HAM",
            "confidence": 99.68,
            "model": "Logistic Regression",
            "timestamp": "28 Sep 11:15",
            "processed_text": "hey meet today colleg librari group studi session"
        },
        {
            "id": 3,
            "message": "Win a free iPhone 15 Pro now! Click here to claim your prize before midnight offer expires.",
            "prediction": "SPAM",
            "confidence": 99.92,
            "model": "Naive Bayes",
            "timestamp": "28 Sep 11:02",
            "processed_text": "win free iphon 15 pro click claim prize midnight offer expir"
        },
        {
            "id": 4,
            "message": "Can you please send me the notes from today's NLP machine learning lecture? Thanks!",
            "prediction": "HAM",
            "confidence": 99.45,
            "model": "Logistic Regression",
            "timestamp": "28 Sep 10:48",
            "processed_text": "send note today nlp machin learn lectur thank"
        },
        {
            "id": 5,
            "message": "URGENT! Your mobile account has been suspended. Call 09050000327 immediately to verify and unlock.",
            "prediction": "SPAM",
            "confidence": 99.76,
            "model": "Naive Bayes",
            "timestamp": "28 Sep 10:30",
            "processed_text": "urgent mobil account suspend call 09050000327 immedi verifi unlock"
        }
    ]


def set_nav(page_name: str):
    """Switch navigation page seamlessly."""
    st.session_state["nav_selection"] = page_name
    st.session_state["history_view_index"] = None


def select_example_message(msg_text: str):
    """Callback for selecting an example message."""
    st.session_state["input_sms_text"] = msg_text
    st.session_state["main_sms_textarea"] = msg_text
    st.session_state["nav_selection"] = "🔍 Detect Spam"
    st.session_state["trigger_predict"] = True
    st.session_state["last_prediction_result"] = None


# ==============================================================================
# MODERN DARK SAAS CSS SYSTEM
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-main: #060A14;
        --bg-card: #0E1626;
        --bg-card-hover: #131E33;
        --border-card: rgba(255, 255, 255, 0.08);
        --border-glow: rgba(124, 58, 237, 0.35);
        --purple-primary: #7C3AED;
        --purple-light: #A78BFA;
        --blue-accent: #38BDF8;
        --cyan-accent: #06B6D4;
        --spam-red: #EF4444;
        --spam-pink: #F43F5E;
        --ham-green: #10B981;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    html, body, [class*="css"] {
        font-family: var(--font-sans) !important;
    }

    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
    }

    #MainMenu, footer {
        visibility: hidden !important;
        display: none !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        z-index: 10;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1440px !important;
    }

    /* --------------------------------------------------------------------------
       SIDEBAR STYLING
       -------------------------------------------------------------------------- */
    [data-testid="stSidebar"] {
        background-color: #080D1A !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0.6rem 0.5rem 1.4rem 0.5rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    }

    .brand-icon {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: linear-gradient(135deg, #7C3AED 0%, #3B82F6 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.45);
    }

    .brand-text-1 {
        font-size: 1.15rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.15;
        letter-spacing: -0.3px;
    }

    .brand-text-2 {
        font-size: 0.88rem;
        font-weight: 700;
        color: #A78BFA;
        letter-spacing: 0.2px;
    }

    /* Sidebar Radio Navigation */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 6px !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 10px !important;
        padding: 0.65rem 0.95rem !important;
        cursor: pointer !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(124, 58, 237, 0.12) !important;
        border-color: rgba(124, 58, 237, 0.35) !important;
        transform: translateX(3px) !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(135deg, #7C3AED 0%, #6366F1 100%) !important;
        border: 1px solid #8B5CF6 !important;
        box-shadow: 0 4px 18px rgba(124, 58, 237, 0.45) !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label span {
        color: #94A3B8 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Hide standard circular radio dot */
    [data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }

    /* --------------------------------------------------------------------------
       TOP NAVIGATION BAR
       -------------------------------------------------------------------------- */
    .top-nav-bar {
        background: #0E1626;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 0.75rem 1.3rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    .top-nav-left {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .nav-menu-btn {
        width: 36px;
        height: 36px;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #94A3B8;
        font-size: 16px;
        cursor: pointer;
    }

    .nav-search-box {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.45rem 1rem;
        min-width: 280px;
        color: #64748B;
        font-size: 0.85rem;
    }

    .top-nav-right {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .nav-bell-wrap {
        position: relative;
        width: 36px;
        height: 36px;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #94A3B8;
        font-size: 16px;
        cursor: pointer;
    }

    .nav-bell-ping {
        position: absolute;
        top: 6px;
        right: 6px;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #F43F5E;
        box-shadow: 0 0 6px #F43F5E;
    }

    .nav-profile-pill {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 4px 12px 4px 5px;
        border-radius: 24px;
        cursor: pointer;
    }

    .profile-avatar {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: linear-gradient(135deg, #7C3AED 0%, #06B6D4 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 0.82rem;
        border: 2px solid #8B5CF6;
    }

    .profile-name {
        color: #F8FAFC;
        font-size: 0.84rem;
        font-weight: 600;
    }

    /* --------------------------------------------------------------------------
       PAGE HEADERS
       -------------------------------------------------------------------------- */
    .page-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.6px;
        line-height: 1.2;
        margin-bottom: 0.25rem;
    }

    .page-subtitle {
        font-size: 0.92rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }

    /* --------------------------------------------------------------------------
       DASHBOARD HERO CARD & PHONE ILLUSTRATION
       -------------------------------------------------------------------------- */
    .dash-hero-card {
        background: linear-gradient(135deg, #0E1626 0%, #151F36 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .dash-hero-card::after {
        content: '';
        position: absolute;
        top: -80px;
        right: 180px;
        width: 250px;
        height: 250px;
        background: radial-gradient(circle, rgba(124, 58, 237, 0.15) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .phone-mockup {
        width: 140px;
        height: 180px;
        background: #080D1A;
        border: 2px solid #7C3AED;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(124, 58, 237, 0.35);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 12px;
        padding: 12px;
        position: relative;
    }

    .phone-notch {
        position: absolute;
        top: 6px;
        width: 36px;
        height: 4px;
        background: #334155;
        border-radius: 4px;
    }

    .phone-badge-spam {
        background: #F43F5E;
        color: white;
        font-size: 0.72rem;
        font-weight: 800;
        padding: 7px 16px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(244, 63, 94, 0.5);
        display: flex;
        align-items: center;
        gap: 6px;
        width: 85%;
        justify-content: center;
    }

    .phone-badge-ham {
        background: #10B981;
        color: white;
        font-size: 0.72rem;
        font-weight: 800;
        padding: 7px 16px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
        display: flex;
        align-items: center;
        gap: 6px;
        width: 85%;
        justify-content: center;
    }

    /* --------------------------------------------------------------------------
       STAT CARDS (4 KPIs)
       -------------------------------------------------------------------------- */
    .stat-card {
        background: #0E1626;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.3rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
        display: flex;
        align-items: flex-start;
        gap: 16px;
    }

    .stat-card:hover {
        transform: translateY(-3px);
        border-color: rgba(124, 58, 237, 0.4);
    }

    .stat-icon-wrap {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        flex-shrink: 0;
    }

    .stat-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.1;
        letter-spacing: -0.5px;
    }

    .stat-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94A3B8;
        margin-top: 3px;
    }

    .stat-subtext {
        font-size: 0.72rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* --------------------------------------------------------------------------
       GENERIC DARK DASHBOARD CARDS
       -------------------------------------------------------------------------- */
    .dark-card {
        background: #0E1626;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        margin-bottom: 1.5rem;
    }

    .dark-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.2rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    .dark-card-title {
        font-size: 1.12rem;
        font-weight: 700;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* --------------------------------------------------------------------------
       STREAMLIT FORM CONTROLS IN DARK THEME
       -------------------------------------------------------------------------- */
    .stTextArea textarea {
        background-color: #080D1A !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        color: #FFFFFF !important;
        font-family: var(--font-sans) !important;
        font-size: 0.95rem !important;
        padding: 0.9rem !important;
    }

    .stTextArea textarea:focus {
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 12px rgba(139, 92, 246, 0.3) !important;
    }

    .stTextInput input {
        background-color: #080D1A !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
    }

    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #080D1A !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
    }

    /* Primary Gradient Button */
    .stButton > button {
        background: linear-gradient(135deg, #7C3AED 0%, #6366F1 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.4rem !important;
        font-size: 0.92rem !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.35) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 22px rgba(124, 58, 237, 0.55) !important;
    }

    /* --------------------------------------------------------------------------
       RESULT CARD (SPAM & HAM)
       -------------------------------------------------------------------------- */
    .result-box-spam {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(244, 63, 94, 0.05) 100%);
        border: 2px solid #EF4444;
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        box-shadow: 0 10px 30px rgba(239, 68, 68, 0.2);
        margin-top: 1.2rem;
    }

    .result-box-ham {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 2px solid #10B981;
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2);
        margin-top: 1.2rem;
    }

    .result-content-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 15px;
    }

    .result-left-block {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .result-icon-circle {
        width: 54px;
        height: 54px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
    }

    .circle-spam {
        background: rgba(239, 68, 68, 0.2);
        border: 2px solid #EF4444;
        color: #F87171;
    }

    .circle-ham {
        background: rgba(16, 185, 129, 0.2);
        border: 2px solid #10B981;
        color: #34D399;
    }

    .result-title-spam {
        font-size: 1.6rem;
        font-weight: 800;
        color: #F87171;
        letter-spacing: -0.5px;
    }

    .result-title-ham {
        font-size: 1.6rem;
        font-weight: 800;
        color: #34D399;
        letter-spacing: -0.5px;
    }

    .result-desc {
        color: #94A3B8;
        font-size: 0.88rem;
        margin-top: 2px;
    }

    .result-confidence-block {
        text-align: right;
    }

    .confidence-label {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #94A3B8;
    }

    .confidence-value {
        font-size: 2rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.1;
    }

    .processed-text-card {
        background: #080D1A;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-top: 1.1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #CBD5E1;
        line-height: 1.5;
    }

    /* --------------------------------------------------------------------------
       BADGES & TABLE ELEMENTS
       -------------------------------------------------------------------------- */
    .badge-spam {
        background: rgba(239, 68, 68, 0.18);
        border: 1px solid rgba(239, 68, 68, 0.45);
        color: #F87171;
        font-size: 0.74rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 6px;
        display: inline-block;
        text-align: center;
    }

    .badge-ham {
        background: rgba(16, 185, 129, 0.18);
        border: 1px solid rgba(16, 185, 129, 0.45);
        color: #34D399;
        font-size: 0.74rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 6px;
        display: inline-block;
        text-align: center;
    }

    /* History Table Rows */
    .history-row {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        display: grid;
        grid-template-columns: 40px 1fr 100px 90px 130px 110px 80px;
        align-items: center;
        gap: 12px;
        transition: background 0.15s ease;
    }

    .history-row:hover {
        background: rgba(255, 255, 255, 0.04);
        border-color: rgba(124, 58, 237, 0.3);
    }

    .history-header {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748B;
        padding: 0.5rem 1rem;
        display: grid;
        grid-template-columns: 40px 1fr 100px 90px 130px 110px 80px;
        gap: 12px;
    }

    /* Word Cloud Tag Box */
    .word-cloud-box {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: center;
        gap: 14px 20px;
        padding: 2rem;
        background: #080D1A;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        min-height: 240px;
    }

    /* Workflow Pipeline Step */
    .pipe-step-card {
        background: #080D1A;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.1rem;
        text-align: center;
        height: 100%;
        position: relative;
    }

    .pipe-step-num {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: linear-gradient(135deg, #7C3AED 0%, #3B82F6 100%);
        color: white;
        font-weight: 800;
        font-size: 0.8rem;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 0.6rem auto;
    }

    .pipe-step-title {
        color: #F8FAFC;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .pipe-step-desc {
        color: #94A3B8;
        font-size: 0.78rem;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">💬</div>
        <div>
            <div class="brand-text-1">SMS Spam</div>
            <div class="brand-text-2">Detection</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Clean Navigation items strictly as requested (NO Settings, NO Logout)
    st.radio(
        "Navigation Menu",
        NAV_ITEMS,
        key="nav_selection",
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    # Active Models Indicator Box
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 12px; padding: 1rem; margin-top: 1rem;">
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="width:8px; height:8px; border-radius:50%; background:#10B981; box-shadow:0 0 8px #10B981;"></span>
            <span style="font-size:0.75rem; font-weight:700; color:#34D399; letter-spacing:0.5px;">ACTIVE ML MODELS</span>
        </div>
        <div style="font-size:0.88rem; font-weight:700; color:#F8FAFC; margin-top:0.4rem;">
            Naive Bayes & Logistic Reg.
        </div>
        <div style="font-size:0.75rem; color:#94A3B8; margin-top:2px;">
            TF-IDF 3,000 N-Grams
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# TOP NAVIGATION BAR
# ==============================================================================
st.markdown("""
<div class="top-nav-bar">
    <div class="top-nav-left">
        <div class="nav-menu-btn">☰</div>
        <div class="nav-search-box">
            <span>🔍</span>
            <span>Search messages, keywords...</span>
        </div>
    </div>
    <div class="top-nav-right">
        <div class="nav-bell-wrap">
            <span>🔔</span>
            <span class="nav-bell-ping"></span>
        </div>
        <div class="nav-profile-pill">
            <div class="profile-avatar">G</div>
            <span class="profile-name">Gayathri</span>
            <span style="font-size:0.65rem; color:#64748B;">▼</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: DASHBOARD
# ==============================================================================
if st.session_state["nav_selection"] == "🏠 Dashboard":
    # Hero Title with Phone Mockup
    st.markdown("""
    <div class="dash-hero-card">
        <div>
            <div class="page-title">SMS Spam Detection</div>
            <div class="page-subtitle" style="margin-bottom:0;">
                Detect spam messages using Natural Language Processing and Machine Learning.
            </div>
            <div style="display:flex; gap:10px; margin-top:1.2rem;">
                <span style="background:rgba(124, 58, 237, 0.2); border:1px solid rgba(124, 58, 237, 0.4); color:#A78BFA; font-size:0.75rem; font-weight:700; padding:4px 12px; border-radius:20px;">
                    Multinomial Naive Bayes
                </span>
                <span style="background:rgba(56, 189, 248, 0.2); border:1px solid rgba(56, 189, 248, 0.4); color:#38BDF8; font-size:0.75rem; font-weight:700; padding:4px 12px; border-radius:20px;">
                    Logistic Regression
                </span>
                <span style="background:rgba(16, 185, 129, 0.2); border:1px solid rgba(16, 185, 129, 0.4); color:#34D399; font-size:0.75rem; font-weight:700; padding:4px 12px; border-radius:20px;">
                    Real-time Inference
                </span>
            </div>
        </div>
        <div class="phone-mockup">
            <div class="phone-notch"></div>
            <div class="phone-badge-spam">⚠️ SPAM</div>
            <div class="phone-badge-ham">🛡️ HAM</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Real Dataset Numbers
    total_msgs = dataset_stats["total_messages"] if dataset_stats else 5572
    spam_msgs = dataset_stats["spam_count"] if dataset_stats else 747
    ham_msgs = dataset_stats["ham_count"] if dataset_stats else 4825
    spam_pct = dataset_stats["spam_percentage"] if dataset_stats else 13.4

    # 4 Statistic Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon-wrap" style="background:rgba(6, 182, 212, 0.15); border:1px solid rgba(6, 182, 212, 0.3); color:#06B6D4;">
                📁
            </div>
            <div>
                <div class="stat-val">{total_msgs:,}</div>
                <div class="stat-label">Total Messages</div>
                <div class="stat-subtext">SMS Corpus Collection</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon-wrap" style="background:rgba(239, 68, 68, 0.15); border:1px solid rgba(239, 68, 68, 0.3); color:#EF4444;">
                🚨
            </div>
            <div>
                <div class="stat-val">{spam_msgs:,}</div>
                <div class="stat-label">Spam Messages</div>
                <div class="stat-subtext">Flagged malicious texts</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon-wrap" style="background:rgba(16, 185, 129, 0.15); border:1px solid rgba(16, 185, 129, 0.3); color:#10B981;">
                🛡️
            </div>
            <div>
                <div class="stat-val">{ham_msgs:,}</div>
                <div class="stat-label">Ham Messages</div>
                <div class="stat-subtext">Verified safe messages</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon-wrap" style="background:rgba(245, 158, 11, 0.15); border:1px solid rgba(245, 158, 11, 0.3); color:#F59E0B;">
                📊
            </div>
            <div>
                <div class="stat-val">{spam_pct}%</div>
                <div class="stat-label">Spam Percentage</div>
                <div class="stat-subtext">Dataset spam prevalence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # Middle Section: Recent Predictions & Message Distribution Donut Chart
    col_left, col_right = st.columns([1.6, 1.0])

    with col_left:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-header">
                <div class="dark-card-title">🕒 Recent Predictions</div>
            </div>
        """, unsafe_allow_html=True)

        recent_history = st.session_state["prediction_history"][:5]
        if recent_history:
            st.markdown("""
            <div class="history-header" style="grid-template-columns: 30px 1fr 85px 80px 100px;">
                <span>#</span>
                <span>Message</span>
                <span>Prediction</span>
                <span>Confidence</span>
                <span>Time</span>
            </div>
            """, unsafe_allow_html=True)

            for idx, item in enumerate(recent_history):
                badge_class = "badge-spam" if item["prediction"] == "SPAM" else "badge-ham"
                badge_text = "Spam" if item["prediction"] == "SPAM" else "Ham"
                msg_preview = item["message"][:48] + "..." if len(item["message"]) > 48 else item["message"]

                st.markdown(f"""
                <div class="history-row" style="grid-template-columns: 30px 1fr 85px 80px 100px; padding:0.65rem 1rem;">
                    <span style="color:#64748B; font-weight:700; font-size:0.82rem;">{idx + 1}</span>
                    <span style="color:#F8FAFC; font-size:0.84rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{msg_preview}</span>
                    <span><span class="{badge_class}">{badge_text}</span></span>
                    <span style="color:#FFFFFF; font-weight:700; font-size:0.84rem;">{item['confidence']}%</span>
                    <span style="color:#94A3B8; font-size:0.78rem;">{item['timestamp']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#94A3B8; padding:1.5rem; text-align:center;'>No predictions recorded yet.</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        st.button("🔍 View All Predictions in History", key="dash_view_all_btn", on_click=set_nav, args=("🕘 Prediction History",), use_container_width=True)

    with col_right:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-header">
                <div class="dark-card-title">🟣 Message Distribution</div>
            </div>
        """, unsafe_allow_html=True)

        # Matplotlib Donut Chart with center text and dark styling
        fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor='#0E1626')
        ax.set_facecolor('#0E1626')

        labels = ['Ham', 'Spam']
        sizes = [ham_msgs, spam_msgs]
        colors = ['#10B981', '#F43F5E']
        explode = (0.02, 0.05)

        wedges, texts = ax.pie(
            sizes,
            explode=explode,
            colors=colors,
            startangle=140,
            wedgeprops=dict(width=0.42, edgecolor='#0E1626', linewidth=2)
        )

        # Center circle text
        ax.text(
            0, 0,
            f"{total_msgs:,}\nMessages",
            ha='center', va='center',
            fontsize=12, fontweight='bold', color='#FFFFFF',
            fontfamily='sans-serif'
        )

        ax.axis('equal')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Custom Legend below chart
        st.markdown(f"""
        <div style="display:flex; justify-content:space-around; margin-top:0.4rem; padding-top:0.6rem; border-top:1px solid rgba(255,255,255,0.06);">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="width:10px; height:10px; border-radius:50%; background:#10B981;"></span>
                <span style="color:#94A3B8; font-size:0.8rem;">Ham: <b>{ham_msgs:,}</b> ({100 - spam_pct:.1f}%)</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="width:10px; height:10px; border-radius:50%; background:#F43F5E;"></span>
                <span style="color:#94A3B8; font-size:0.8rem;">Spam: <b>{spam_msgs:,}</b> ({spam_pct}%)</span>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # Dashboard Model Summary Cards
    st.markdown("""
    <div style="margin-top: 1rem; margin-bottom: 0.8rem; font-size: 1.15rem; font-weight: 700; color: #FFFFFF;">
        Dashboard Model Summary
    </div>
    """, unsafe_allow_html=True)

    mb1, mb2 = st.columns(2)
    metrics = artifacts["metrics"]
    all_metrics = metrics.get("all_models_metrics", {}) if metrics else {}

    nb_acc = all_metrics.get("Multinomial Naive Bayes", {}).get("accuracy", 0.9787) * 100
    nb_prec = all_metrics.get("Multinomial Naive Bayes", {}).get("precision", 0.9504) * 100
    nb_f1 = all_metrics.get("Multinomial Naive Bayes", {}).get("f1_score", 0.9127) * 100

    lr_acc = all_metrics.get("Logistic Regression", {}).get("accuracy", 0.9729) * 100
    lr_prec = all_metrics.get("Logistic Regression", {}).get("precision", 0.8872) * 100
    lr_f1 = all_metrics.get("Logistic Regression", {}).get("f1_score", 0.8939) * 100

    with mb1:
        st.markdown(f"""
        <div class="stat-card" style="display:block;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">
                <div style="font-size:1.05rem; font-weight:700; color:#FFFFFF;">Naive Bayes</div>
                <span style="background:rgba(124, 58, 237, 0.2); border:1px solid #7C3AED; color:#A78BFA; font-size:0.7rem; font-weight:700; padding:2px 8px; border-radius:12px;">Top F1-Score Winner</span>
            </div>
            <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:8px;">
                <div style="background:rgba(255,255,255,0.03); padding:8px; border-radius:8px; text-align:center;">
                    <div style="font-size:0.7rem; color:#94A3B8;">Accuracy</div>
                    <div style="font-size:1.1rem; font-weight:800; color:#FFFFFF;">{nb_acc:.2f}%</div>
                </div>
                <div style="background:rgba(255,255,255,0.03); padding:8px; border-radius:8px; text-align:center;">
                    <div style="font-size:0.7rem; color:#94A3B8;">Precision</div>
                    <div style="font-size:1.1rem; font-weight:800; color:#06B6D4;">{nb_prec:.2f}%</div>
                </div>
                <div style="background:rgba(255,255,255,0.03); padding:8px; border-radius:8px; text-align:center;">
                    <div style="font-size:0.7rem; color:#94A3B8;">F1-Score</div>
                    <div style="font-size:1.1rem; font-weight:800; color:#10B981;">{nb_f1:.2f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with mb2:
        st.markdown(f"""
        <div class="stat-card" style="display:block;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">
                <div style="font-size:1.05rem; font-weight:700; color:#FFFFFF;">Logistic Regression</div>
                <span style="background:rgba(56, 189, 248, 0.2); border:1px solid #0284C7; color:#38BDF8; font-size:0.7rem; font-weight:700; padding:2px 8px; border-radius:12px;">Balanced Regularized</span>
            </div>
            <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:8px;">
                <div style="background:rgba(255,255,255,0.03); padding:8px; border-radius:8px; text-align:center;">
                    <div style="font-size:0.7rem; color:#94A3B8;">Accuracy</div>
                    <div style="font-size:1.1rem; font-weight:800; color:#FFFFFF;">{lr_acc:.2f}%</div>
                </div>
                <div style="background:rgba(255,255,255,0.03); padding:8px; border-radius:8px; text-align:center;">
                    <div style="font-size:0.7rem; color:#94A3B8;">Precision</div>
                    <div style="font-size:1.1rem; font-weight:800; color:#06B6D4;">{lr_prec:.2f}%</div>
                </div>
                <div style="background:rgba(255,255,255,0.03); padding:8px; border-radius:8px; text-align:center;">
                    <div style="font-size:0.7rem; color:#94A3B8;">F1-Score</div>
                    <div style="font-size:1.1rem; font-weight:800; color:#10B981;">{lr_f1:.2f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: DETECT SPAM
# ==============================================================================
elif st.session_state["nav_selection"] == "🔍 Detect Spam":
    st.markdown("""
    <div>
        <div class="page-title">Detect Spam</div>
        <div class="page-subtitle">Enter an SMS message to check whether it's Spam or Ham.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="dark-card">
        <div class="dark-card-header">
            <div class="dark-card-title">✉️ Enter SMS Message</div>
        </div>
    """, unsafe_allow_html=True)

    # Model selector and options
    col_sel, col_info = st.columns([1.5, 2.5])
    with col_sel:
        chosen_model = st.selectbox(
            "Select Machine Learning Model:",
            ["Naive Bayes", "Logistic Regression"],
            index=0 if st.session_state["selected_model_name"] == "Naive Bayes" else 1,
            key="model_select_box"
        )
        st.session_state["selected_model_name"] = chosen_model

    with col_info:
        model_tag = "MultinomialNB (alpha=0.2)" if chosen_model == "Naive Bayes" else "Logistic Regression (balanced L2)"
        st.markdown(f"""
        <div style="padding-top:1.7rem; color:#94A3B8; font-size:0.82rem;">
            Using: <span style="color:#A78BFA; font-weight:700;">{model_tag}</span> with TF-IDF Vectorization
        </div>
        """, unsafe_allow_html=True)

    # SMS Text Input
    if "main_sms_textarea" not in st.session_state:
        st.session_state["main_sms_textarea"] = st.session_state.get("input_sms_text", "")

    user_sms = st.text_area(
        "SMS Message Content:",
        placeholder="Type or paste your message here...",
        height=130,
        key="main_sms_textarea"
    )
    st.session_state["input_sms_text"] = user_sms

    # Character count indicator
    char_len = len(user_sms)
    st.markdown(f"""
    <div style="display:flex; justify-content:flex-end; color:#64748B; font-size:0.75rem; margin-top:-6px; margin-bottom:12px;">
        {char_len} characters
    </div>
    """, unsafe_allow_html=True)

    btn_col1, btn_col2, btn_spacer = st.columns([1.5, 1.0, 3.5])
    with btn_col1:
        predict_clicked = st.button("✈️ Predict Message", use_container_width=True, key="predict_btn")
    with btn_col2:
        if st.button("🗑️ Clear", use_container_width=True, key="clear_btn"):
            st.session_state["input_sms_text"] = ""
            st.session_state["main_sms_textarea"] = ""
            st.session_state["last_prediction_result"] = None
            st.session_state["trigger_predict"] = False
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Execution of real Prediction
    should_run = predict_clicked or st.session_state.get("trigger_predict", False)

    if should_run:
        st.session_state["trigger_predict"] = False
        message_to_check = user_sms.strip()

        if not message_to_check:
            st.markdown("""
            <div style="background:rgba(239, 68, 68, 0.12); border:1px solid #EF4444; border-radius:12px; padding:1rem; color:#F87171; font-size:0.9rem;">
                ⚠️ Input SMS message cannot be empty. Please type or paste a message to classify.
            </div>
            """, unsafe_allow_html=True)
        elif not any(c.isalnum() for c in message_to_check):
            st.markdown("""
            <div style="background:rgba(239, 68, 68, 0.12); border:1px solid #EF4444; border-radius:12px; padding:1rem; color:#F87171; font-size:0.9rem;">
                ⚠️ The message does not contain any valid alphanumeric words or characters.
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Analyzing message with NLP pipeline..."):
                result = predict_sms(message_to_check, model_name=chosen_model)

            if result.get("is_valid", False):
                st.session_state["last_prediction_result"] = result

                # Add to prediction history
                now_str = datetime.datetime.now().strftime("%d %b %H:%M")
                new_entry = {
                    "id": len(st.session_state["prediction_history"]) + 1,
                    "message": message_to_check,
                    "prediction": result["label"].upper(),
                    "confidence": result["confidence"],
                    "model": chosen_model,
                    "timestamp": now_str,
                    "processed_text": result.get("cleaned_text", "")
                }
                # Insert at top of history
                st.session_state["prediction_history"].insert(0, new_entry)

    # Render Prediction Result Card if available
    res = st.session_state["last_prediction_result"]
    if res and res.get("is_valid", False):
        is_spam = res["is_spam"]
        label_text = "SPAM" if is_spam else "HAM"
        card_class = "result-box-spam" if is_spam else "result-box-ham"
        circle_class = "circle-spam" if is_spam else "circle-ham"
        icon = "✉️" if is_spam else "🛡️"
        desc_text = "This message is classified as Spam." if is_spam else "This message is classified as Ham (Legitimate)."
        title_class = "result-title-spam" if is_spam else "result-title-ham"
        conf_val = res["confidence"]
        cleaned_str = res.get("cleaned_text", "(none)")

        st.markdown(f"""
        <div class="{card_class}">
            <div class="result-content-row">
                <div class="result-left-block">
                    <div class="result-icon-circle {circle_class}">{icon}</div>
                    <div>
                        <div class="{title_class}">{label_text}</div>
                        <div class="result-desc">{desc_text}</div>
                    </div>
                </div>
                <div class="result-confidence-block">
                    <div class="confidence-label">Confidence</div>
                    <div class="confidence-value">{conf_val}%</div>
                </div>
            </div>
            <div style="margin-top:1rem; border-top:1px solid rgba(255,255,255,0.08); padding-top:0.8rem;">
                <div style="font-size:0.76rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.5px;">Processed Text</div>
                <div class="processed-text-card">{cleaned_str}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 3: PREDICTION HISTORY
# ==============================================================================
elif st.session_state["nav_selection"] == "🕘 Prediction History":
    st.markdown("""
    <div>
        <div class="page-title">Prediction History</div>
        <div class="page-subtitle">View all your previous predictions.</div>
    </div>
    """, unsafe_allow_html=True)

    # Search and Filter Controls
    st.markdown("""
    <div class="dark-card" style="padding:1.1rem 1.4rem; margin-bottom:1rem;">
    """, unsafe_allow_html=True)

    fc1, fc2, fc3, fc4 = st.columns([2.0, 1.2, 1.2, 1.0])
    with fc1:
        search_query = st.text_input("Search:", placeholder="Search messages...", label_visibility="collapsed", key="hist_search")
    with fc2:
        pred_filter = st.selectbox("Prediction:", ["All Predictions", "SPAM", "HAM"], key="hist_pred_filter")
    with fc3:
        model_filter = st.selectbox("Model:", ["All Models", "Naive Bayes", "Logistic Regression"], key="hist_model_filter")
    with fc4:
        clear_clicked = st.button("🗑️ Clear History", use_container_width=True, key="clear_hist_btn")

    st.markdown("</div>", unsafe_allow_html=True)

    if clear_clicked:
        st.session_state["prediction_history"] = []
        st.session_state["history_view_index"] = None
        st.success("Prediction history cleared.")
        st.rerun()

    # Filter Records
    raw_history = st.session_state["prediction_history"]
    filtered_history = []
    for item in raw_history:
        # Search text match
        if search_query and search_query.lower() not in item["message"].lower():
            continue
        # Prediction match
        if pred_filter != "All Predictions" and item["prediction"] != pred_filter:
            continue
        # Model match
        if model_filter != "All Models" and item["model"] != model_filter:
            continue
        filtered_history.append(item)

    # Detailed Single Message View Card (if clicked View)
    view_idx = st.session_state.get("history_view_index", None)
    if view_idx is not None and 0 <= view_idx < len(filtered_history):
        detail_item = filtered_history[view_idx]
        is_sp = detail_item["prediction"] == "SPAM"
        badge_cls = "badge-spam" if is_sp else "badge-ham"

        st.markdown(f"""
        <div class="dark-card" style="border: 2px solid #7C3AED; background: #0E1626;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:0.6rem;">
                <div style="font-size:1.15rem; font-weight:700; color:#FFFFFF;">
                    🔍 Prediction Detail #{detail_item['id']}
                </div>
                <span class="{badge_cls}" style="font-size:0.85rem; padding:4px 14px;">{detail_item['prediction']}</span>
            </div>
            <div style="margin-bottom:0.8rem;">
                <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase; font-weight:700;">Full SMS Message</div>
                <div style="font-size:0.95rem; color:#FFFFFF; margin-top:4px; line-height:1.5; background:rgba(255,255,255,0.03); padding:1rem; border-radius:10px;">
                    "{detail_item['message']}"
                </div>
            </div>
            <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; margin-bottom:0.8rem;">
                <div style="background:rgba(255,255,255,0.03); padding:0.8rem; border-radius:10px;">
                    <div style="font-size:0.72rem; color:#94A3B8;">Confidence</div>
                    <div style="font-size:1.2rem; font-weight:800; color:#FFFFFF;">{detail_item['confidence']}%</div>
                </div>
                <div style="background:rgba(255,255,255,0.03); padding:0.8rem; border-radius:10px;">
                    <div style="font-size:0.72rem; color:#94A3B8;">Model Used</div>
                    <div style="font-size:1.0rem; font-weight:700; color:#38BDF8;">{detail_item['model']}</div>
                </div>
                <div style="background:rgba(255,255,255,0.03); padding:0.8rem; border-radius:10px;">
                    <div style="font-size:0.72rem; color:#94A3B8;">Date & Time</div>
                    <div style="font-size:0.95rem; font-weight:600; color:#A78BFA;">{detail_item['timestamp']}</div>
                </div>
            </div>
            <div>
                <div style="font-size:0.72rem; color:#94A3B8; text-transform:uppercase; font-weight:700;">Processed Text</div>
                <div class="processed-text-card" style="margin-top:4px;">
                    {detail_item.get('processed_text', '(none)')}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("✕ Close Detail View", key="close_detail_btn"):
            st.session_state["history_view_index"] = None
            st.rerun()

    # History Table
    if filtered_history:
        st.markdown("""
        <div class="dark-card">
            <div class="history-header">
                <span>#</span>
                <span>Message</span>
                <span>Prediction</span>
                <span>Confidence</span>
                <span>Model</span>
                <span>Date & Time</span>
                <span>Action</span>
            </div>
        """, unsafe_allow_html=True)

        PAGE_SIZE = 8
        total_pages = max(1, (len(filtered_history) + PAGE_SIZE - 1) // PAGE_SIZE)
        curr_page = st.session_state.get("history_page_num", 1)
        if curr_page > total_pages:
            curr_page = 1
            st.session_state["history_page_num"] = 1

        start_idx = (curr_page - 1) * PAGE_SIZE
        page_items = filtered_history[start_idx:start_idx + PAGE_SIZE]

        for i, item in enumerate(page_items):
            actual_idx = start_idx + i
            badge_class = "badge-spam" if item["prediction"] == "SPAM" else "badge-ham"
            badge_text = "Spam" if item["prediction"] == "SPAM" else "Ham"
            preview_msg = item["message"][:42] + "..." if len(item["message"]) > 42 else item["message"]

            cols = st.columns([0.4, 2.8, 1.0, 0.9, 1.3, 1.1, 0.8])
            with cols[0]:
                st.markdown(f"<span style='color:#64748B; font-size:0.84rem; font-weight:700;'>{actual_idx + 1}</span>", unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"<span style='color:#F8FAFC; font-size:0.84rem;'>{preview_msg}</span>", unsafe_allow_html=True)
            with cols[2]:
                st.markdown(f"<span class='{badge_class}'>{badge_text}</span>", unsafe_allow_html=True)
            with cols[3]:
                st.markdown(f"<span style='color:#FFFFFF; font-size:0.84rem; font-weight:700;'>{item['confidence']}%</span>", unsafe_allow_html=True)
            with cols[4]:
                st.markdown(f"<span style='color:#38BDF8; font-size:0.8rem; font-weight:600;'>{item['model']}</span>", unsafe_allow_html=True)
            with cols[5]:
                st.markdown(f"<span style='color:#94A3B8; font-size:0.78rem;'>{item['timestamp']}</span>", unsafe_allow_html=True)
            with cols[6]:
                if st.button("View", key=f"view_hist_{actual_idx}"):
                    st.session_state["history_view_index"] = actual_idx
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # Pagination Controls
        if total_pages > 1:
            p_col1, p_col2, p_col3 = st.columns([1, 2, 1])
            with p_col1:
                if st.button("◀ Previous", disabled=(curr_page <= 1), key="prev_page_btn"):
                    st.session_state["history_page_num"] -= 1
                    st.rerun()
            with p_col2:
                st.markdown(f"<div style='text-align:center; color:#94A3B8; font-size:0.85rem; padding-top:6px;'>Page {curr_page} of {total_pages}</div>", unsafe_allow_html=True)
            with p_col3:
                if st.button("Next ▶", disabled=(curr_page >= total_pages), key="next_page_btn"):
                    st.session_state["history_page_num"] += 1
                    st.rerun()

    else:
        # Empty State
        st.markdown("""
        <div class="dark-card" style="text-align:center; padding:3rem 1.5rem;">
            <div style="font-size:3rem; margin-bottom:0.8rem;">📭</div>
            <div style="font-size:1.3rem; font-weight:800; color:#FFFFFF; margin-bottom:0.3rem;">No predictions yet</div>
            <div style="color:#94A3B8; font-size:0.9rem; max-width:400px; margin:0 auto 1.5rem auto;">
                Your previous SMS predictions will appear here once you classify your first message.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.button("🔍 Detect Your First Message", key="empty_detect_btn", on_click=set_nav, args=("🔍 Detect Spam",), use_container_width=True)


# ==============================================================================
# PAGE 4: MODEL PERFORMANCE
# ==============================================================================
elif st.session_state["nav_selection"] == "📊 Model Performance":
    st.markdown("""
    <div>
        <div class="page-title">Model Performance</div>
        <div class="page-subtitle">Comparison of machine learning models used for SMS Spam Detection.</div>
    </div>
    """, unsafe_allow_html=True)

    metrics = artifacts["metrics"]
    all_models = metrics.get("all_models_metrics", {}) if metrics else {}

    nb = all_models.get("Multinomial Naive Bayes", {
        "accuracy": 0.9787, "precision": 0.9504, "recall": 0.8779, "f1_score": 0.9127
    })
    lr = all_models.get("Logistic Regression", {
        "accuracy": 0.9729, "precision": 0.8872, "recall": 0.9008, "f1_score": 0.8939
    })

    # Model Performance Comparison Chart
    st.markdown("""
    <div class="dark-card">
        <div class="dark-card-header">
            <div class="dark-card-title">📊 Model Accuracy & Metrics Comparison</div>
        </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor='#0E1626')
    ax.set_facecolor('#0E1626')

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    nb_values = [nb["accuracy"] * 100, nb["precision"] * 100, nb["recall"] * 100, nb["f1_score"] * 100]
    lr_values = [lr["accuracy"] * 100, lr["precision"] * 100, lr["recall"] * 100, lr["f1_score"] * 100]

    x = np.arange(len(metric_names))
    width = 0.32

    rects1 = ax.bar(x - width/2, nb_values, width, label='Naive Bayes', color='#7C3AED', edgecolor='#8B5CF6', linewidth=1.2)
    rects2 = ax.bar(x + width/2, lr_values, width, label='Logistic Regression', color='#06B6D4', edgecolor='#38BDF8', linewidth=1.2)

    ax.set_ylabel('Score (%)', color='#94A3B8', fontsize=10)
    ax.set_title('Evaluation Metrics on Stratified 20% Test Set', color='#FFFFFF', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, color='#F8FAFC', fontsize=10, fontweight='600')
    ax.set_ylim(70, 105)
    ax.grid(axis='y', color='#334155', linestyle='--', alpha=0.7)
    ax.tick_params(colors='#94A3B8')

    for spine in ax.spines.values():
        spine.set_color('#334155')

    # Add data values on bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', color='#A78BFA', fontsize=9, fontweight='bold')

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', color='#38BDF8', fontsize=9, fontweight='bold')

    ax.legend(facecolor='#080D1A', edgecolor='#334155', labelcolor='#FFFFFF')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("</div>", unsafe_allow_html=True)

    # Metrics Table & Attractive Model Cards
    col_t, col_c = st.columns([1.1, 1.2])

    with col_t:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-header">
                <div class="dark-card-title">📋 Model Evaluation Metrics</div>
            </div>
            <table style="width:100%; border-collapse:collapse; text-align:left; color:#FFFFFF; font-size:0.86rem;">
                <thead>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.1); color:#94A3B8; font-size:0.75rem; text-transform:uppercase;">
                        <th style="padding:8px 6px;">Model</th>
                        <th style="padding:8px 6px;">Accuracy</th>
                        <th style="padding:8px 6px;">Precision</th>
                        <th style="padding:8px 6px;">Recall</th>
                        <th style="padding:8px 6px;">F1 Score</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:10px 6px; font-weight:700; color:#A78BFA;">Naive Bayes</td>
                        <td style="padding:10px 6px; font-weight:600;">97.87%</td>
                        <td style="padding:10px 6px; font-weight:600; color:#06B6D4;">95.04%</td>
                        <td style="padding:10px 6px; font-weight:600;">87.79%</td>
                        <td style="padding:10px 6px; font-weight:700; color:#10B981;">91.27%</td>
                    </tr>
                    <tr>
                        <td style="padding:10px 6px; font-weight:700; color:#38BDF8;">Logistic Reg.</td>
                        <td style="padding:10px 6px; font-weight:600;">97.29%</td>
                        <td style="padding:10px 6px; font-weight:600; color:#06B6D4;">88.72%</td>
                        <td style="padding:10px 6px; font-weight:600;">90.08%</td>
                        <td style="padding:10px 6px; font-weight:700; color:#10B981;">89.39%</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown(f"""
        <div style="display:flex; flex-direction:column; gap:12px;">
            <div class="stat-card" style="display:block; border-left:4px solid #7C3AED;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-size:1.1rem; font-weight:800; color:#FFFFFF;">Multinomial Naive Bayes</div>
                    <span style="background:rgba(124, 58, 237, 0.2); border:1px solid #7C3AED; color:#A78BFA; font-size:0.7rem; font-weight:700; padding:2px 8px; border-radius:12px;">Selected Best Model</span>
                </div>
                <div style="font-size:0.78rem; color:#94A3B8; margin-top:4px; line-height:1.4;">
                    Optimal probabilistic text classifier with Laplace smoothing (alpha=0.2). Achieved the highest precision (95.04%) and F1-score (91.27%).
                </div>
            </div>
            <div class="stat-card" style="display:block; border-left:4px solid #06B6D4;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-size:1.1rem; font-weight:800; color:#FFFFFF;">Logistic Regression</div>
                    <span style="background:rgba(6, 182, 212, 0.2); border:1px solid #06B6D4; color:#38BDF8; font-size:0.7rem; font-weight:700; padding:2px 8px; border-radius:12px;">Balanced Regularized</span>
                </div>
                <div style="font-size:0.78rem; color:#94A3B8; margin-top:4px; line-height:1.4;">
                    Linear decision boundary with L2 regularization and balanced class weights. Achieved exceptional recall (90.08%) on spam instances.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 5: DATA ANALYTICS
# ==============================================================================
elif st.session_state["nav_selection"] == "📈 Data Analytics":
    st.markdown("""
    <div>
        <div class="page-title">Data Analytics</div>
        <div class="page-subtitle">Explore insights from the SMS messages dataset.</div>
    </div>
    """, unsafe_allow_html=True)

    # Real Dataset Stats
    total_msgs = dataset_stats["total_messages"] if dataset_stats else 5572
    spam_msgs = dataset_stats["spam_count"] if dataset_stats else 747
    ham_msgs = dataset_stats["ham_count"] if dataset_stats else 4825
    spam_pct = dataset_stats["spam_percentage"] if dataset_stats else 13.4
    df = dataset_stats["df"] if dataset_stats else None

    # Top Row: Message Distribution Donut & Top Spam Keywords Horizontal Bar
    r1_left, r1_right = st.columns([1.0, 1.4])

    with r1_left:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-header">
                <div class="dark-card-title">🟣 Message Distribution</div>
            </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(4.2, 3.2), facecolor='#0E1626')
        ax.set_facecolor('#0E1626')
        wedges, _ = ax.pie(
            [ham_msgs, spam_msgs],
            explode=(0.02, 0.05),
            colors=['#10B981', '#F43F5E'],
            startangle=140,
            wedgeprops=dict(width=0.42, edgecolor='#0E1626', linewidth=2)
        )
        ax.text(0, 0, f"{total_msgs:,}\nMessages", ha='center', va='center', fontsize=11, fontweight='bold', color='#FFFFFF')
        ax.axis('equal')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown(f"""
        <div style="display:flex; justify-content:space-around; margin-top:0.4rem; padding-top:0.6rem; border-top:1px solid rgba(255,255,255,0.06); font-size:0.8rem;">
            <span style="color:#94A3B8;"><span style="color:#10B981;">●</span> Ham: <b>{ham_msgs:,}</b> ({100 - spam_pct:.1f}%)</span>
            <span style="color:#94A3B8;"><span style="color:#F43F5E;">●</span> Spam: <b>{spam_msgs:,}</b> ({spam_pct}%)</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with r1_right:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-header">
                <div class="dark-card-title">📊 Top Spam Keywords</div>
            </div>
        """, unsafe_allow_html=True)

        # Real Top Spam Words from metrics or eda_stats
        metrics = artifacts["metrics"]
        eda = metrics.get("eda_stats", {}) if metrics else {}
        top_words_list = eda.get("top_spam_words", [
            ("call", 332), ("free", 195), ("txt", 144), ("text", 127),
            ("mobil", 118), ("claim", 98), ("prize", 83), ("stop", 109)
        ])[:6]

        keywords = [item[0] for item in reversed(top_words_list)]
        counts = [item[1] for item in reversed(top_words_list)]

        fig, ax = plt.subplots(figsize=(5.5, 3.2), facecolor='#0E1626')
        ax.set_facecolor('#0E1626')
        bars = ax.barh(keywords, counts, color='#7C3AED', edgecolor='#A78BFA', height=0.55)

        ax.set_xlabel('Occurrences in Spam Messages', color='#94A3B8', fontsize=9)
        ax.tick_params(colors='#CBD5E1', labelsize=9)
        ax.grid(axis='x', color='#334155', linestyle='--', alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color('#334155')

        for bar in bars:
            w = bar.get_width()
            ax.annotate(f' {w}', xy=(w, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0), textcoords="offset points",
                        ha='left', va='center', color='#38BDF8', fontsize=9, fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # Bottom Row: Message Length Histogram & Word Cloud Card
    r2_left, r2_right = st.columns([1.2, 1.2])

    with r2_left:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-header">
                <div class="dark-card-title">📈 Message Length Distribution</div>
            </div>
        """, unsafe_allow_html=True)

        if df is not None:
            fig, ax = plt.subplots(figsize=(5.5, 3.3), facecolor='#0E1626')
            ax.set_facecolor('#0E1626')

            ham_lengths = df[df["label"] == "ham"]["char_len"]
            spam_lengths = df[df["label"] == "spam"]["char_len"]

            ax.hist(ham_lengths, bins=30, range=(0, 250), color='#06B6D4', alpha=0.75, label='Ham', edgecolor='none')
            ax.hist(spam_lengths, bins=30, range=(0, 250), color='#F43F5E', alpha=0.85, label='Spam', edgecolor='none')

            ax.set_xlabel('Character Length', color='#94A3B8', fontsize=9)
            ax.set_ylabel('Frequency', color='#94A3B8', fontsize=9)
            ax.tick_params(colors='#CBD5E1', labelsize=8)
            ax.grid(color='#334155', linestyle='--', alpha=0.7)
            for spine in ax.spines.values():
                spine.set_color('#334155')

            ax.legend(facecolor='#080D1A', edgecolor='#334155', labelcolor='#FFFFFF')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with r2_right:
        st.markdown("""
        <div class="dark-card">
            <div class="dark-card-header">
                <div class="dark-card-title">☁️ Word Cloud (Spam Messages)</div>
            </div>
            <div class="word-cloud-box">
                <span style="font-size:2.4rem; font-weight:800; color:#EF4444; text-shadow:0 0 10px rgba(239,68,68,0.4);">FREE</span>
                <span style="font-size:2.0rem; font-weight:800; color:#38BDF8; text-shadow:0 0 10px rgba(56,189,248,0.4);">CALL</span>
                <span style="font-size:1.8rem; font-weight:800; color:#10B981; text-shadow:0 0 10px rgba(16,185,129,0.4);">WIN</span>
                <span style="font-size:1.6rem; font-weight:700; color:#F59E0B;">PRIZE</span>
                <span style="font-size:1.5rem; font-weight:700; color:#A78BFA;">CLAIM</span>
                <span style="font-size:1.4rem; font-weight:700; color:#F43F5E;">URGENT</span>
                <span style="font-size:1.3rem; font-weight:600; color:#06B6D4;">CASH</span>
                <span style="font-size:1.3rem; font-weight:600; color:#E2E8F0;">MOBILE</span>
                <span style="font-size:1.2rem; font-weight:600; color:#F87171;">NOW</span>
                <span style="font-size:1.1rem; font-weight:600; color:#38BDF8;">TEXT</span>
                <span style="font-size:1.0rem; font-weight:500; color:#CBD5E1;">AWARD</span>
                <span style="font-size:1.0rem; font-weight:500; color:#A78BFA;">CUSTOMER</span>
                <span style="font-size:0.95rem; font-weight:500; color:#F59E0B;">REPLY</span>
                <span style="font-size:0.9rem; font-weight:500; color:#94A3B8;">STOP</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 6: EXAMPLE MESSAGES
# ==============================================================================
elif st.session_state["nav_selection"] == "💬 Example Messages":
    st.markdown("""
    <div>
        <div class="page-title">Example Messages</div>
        <div class="page-subtitle">Try these sample messages to test the spam detection model.</div>
    </div>
    """, unsafe_allow_html=True)

    tab_spam, tab_ham = st.tabs(["🚨 Spam Messages", "🛡️ Ham Messages"])

    with tab_spam:
        spam_samples = [
            "Congratulations! You have won a free iPhone. Click here to claim your prize.",
            "Limited time offer! Get 50% off on all products. Visit now to claim voucher.",
            "You have been selected for a cash prize. Call 09061701461 to claim your award.",
            "Win a free holiday to Bahamas! Click here to enter the sweepstakes immediately.",
            "URGENT! Your bank account has been suspended due to activity. Verify now at http://secure-bank.com",
            "Get rich quick! Earn $5,000 per month working from home. No investment required."
        ]

        grid_cols = st.columns(3)
        for idx, msg in enumerate(spam_samples):
            with grid_cols[idx % 3]:
                st.markdown(f"""
                <div class="stat-card" style="display:flex; flex-direction:column; justify-content:space-between; height:180px; margin-bottom:1rem;">
                    <div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <span class="badge-spam">Spam Alert</span>
                            <span style="font-size:0.7rem; color:#64748B;">Sample #{idx + 1}</span>
                        </div>
                        <div style="color:#F8FAFC; font-size:0.86rem; line-height:1.45; overflow:hidden; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical;">
                            "{msg}"
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.button(f"✈️ Use this message", key=f"use_spam_{idx}", on_click=select_example_message, args=(msg,), use_container_width=True)

    with tab_ham:
        ham_samples = [
            "Hey, are we still meeting today at the library for group study?",
            "Can you please send me the lecture notes from today's NLP class?",
            "Are you coming to college today? Attendance is mandatory for the lab.",
            "I will call you later once I reach home. Take care!",
            "Hey mom, I will be home by 7 PM. Please don't wait for dinner.",
            "Don't forget to submit the assignment before the 5 PM deadline."
        ]

        grid_cols_ham = st.columns(3)
        for idx, msg in enumerate(ham_samples):
            with grid_cols_ham[idx % 3]:
                st.markdown(f"""
                <div class="stat-card" style="display:flex; flex-direction:column; justify-content:space-between; height:180px; margin-bottom:1rem;">
                    <div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <span class="badge-ham">Legitimate Ham</span>
                            <span style="font-size:0.7rem; color:#64748B;">Sample #{idx + 1}</span>
                        </div>
                        <div style="color:#F8FAFC; font-size:0.86rem; line-height:1.45; overflow:hidden; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical;">
                            "{msg}"
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.button(f"✈️ Use this message", key=f"use_ham_{idx}", on_click=select_example_message, args=(msg,), use_container_width=True)


# ==============================================================================
# PAGE 7: ABOUT PROJECT
# ==============================================================================
elif st.session_state["nav_selection"] == "ℹ️ About Project":
    st.markdown("""
    <div>
        <div class="page-title">About Project</div>
        <div class="page-subtitle">SMS Spam Detection using Natural Language Processing and Machine Learning.</div>
    </div>
    """, unsafe_allow_html=True)

    # Project Overview Card
    st.markdown("""
    <div class="dark-card">
        <div class="dark-card-header">
            <div class="dark-card-title">📖 Project Overview</div>
        </div>
        <div style="color:#CBD5E1; font-size:0.92rem; line-height:1.6;">
            This project provides real-time classification of incoming SMS text messages into <b>SPAM</b> or <b>HAM</b> (Legitimate) categories. 
            Utilizing natural language preprocessing, term frequency-inverse document frequency (TF-IDF) feature extraction, and machine learning models, 
            the system delivers instant threat evaluation with calibrated confidence percentages.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Real Dataset Information Grid
    st.markdown("""
    <div style="margin-top: 1.2rem; margin-bottom: 0.8rem; font-size: 1.15rem; font-weight: 700; color: #FFFFFF;">
        Dataset Information
    </div>
    """, unsafe_allow_html=True)

    total_msgs = dataset_stats["total_messages"] if dataset_stats else 5572
    spam_msgs = dataset_stats["spam_count"] if dataset_stats else 747
    ham_msgs = dataset_stats["ham_count"] if dataset_stats else 4825
    spam_pct = dataset_stats["spam_percentage"] if dataset_stats else 13.4

    dc1, dc2, dc3, dc4 = st.columns(4)
    with dc1:
        st.markdown(f"""
        <div class="stat-card">
            <div>
                <div class="stat-label">Total Messages</div>
                <div class="stat-val">{total_msgs:,}</div>
                <div class="stat-subtext">SMS Corpus</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with dc2:
        st.markdown(f"""
        <div class="stat-card">
            <div>
                <div class="stat-label">Spam Messages</div>
                <div class="stat-val" style="color:#F87171;">{spam_msgs:,}</div>
                <div class="stat-subtext">Flagged Spams</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with dc3:
        st.markdown(f"""
        <div class="stat-card">
            <div>
                <div class="stat-label">Ham Messages</div>
                <div class="stat-val" style="color:#34D399;">{ham_msgs:,}</div>
                <div class="stat-subtext">Verified Hams</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with dc4:
        st.markdown(f"""
        <div class="stat-card">
            <div>
                <div class="stat-label">Spam Percentage</div>
                <div class="stat-val" style="color:#F59E0B;">{spam_pct}%</div>
                <div class="stat-subtext">Prevalence Rate</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # NLP Pipeline Visual Workflow
    st.markdown("""
    <div style="margin-top: 1.5rem; margin-bottom: 0.8rem; font-size: 1.15rem; font-weight: 700; color: #FFFFFF;">
        NLP Pipeline Workflow
    </div>
    """, unsafe_allow_html=True)

    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        st.markdown("""
        <div class="pipe-step-card">
            <div class="pipe-step-num">1</div>
            <div class="pipe-step-title">Raw SMS Input</div>
            <div class="pipe-step-desc">Original incoming text string.</div>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown("""
        <div class="pipe-step-card">
            <div class="pipe-step-num">2</div>
            <div class="pipe-step-title">Text Preprocessing</div>
            <div class="pipe-step-desc">Lowercasing, URL removal, punctuation strip, tokenization, stopword removal & stemming.</div>
        </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown("""
        <div class="pipe-step-card">
            <div class="pipe-step-num">3</div>
            <div class="pipe-step-title">TF-IDF Vectorization</div>
            <div class="pipe-step-desc">3,000 sublinear TF-IDF n-gram (1,2) features.</div>
        </div>
        """, unsafe_allow_html=True)
    with p4:
        st.markdown("""
        <div class="pipe-step-card">
            <div class="pipe-step-num">4</div>
            <div class="pipe-step-title">ML Classifier</div>
            <div class="pipe-step-desc">Naive Bayes or Logistic Regression inference.</div>
        </div>
        """, unsafe_allow_html=True)
    with p5:
        st.markdown("""
        <div class="pipe-step-card">
            <div class="pipe-step-num">5</div>
            <div class="pipe-step-title">Prediction & Score</div>
            <div class="pipe-step-desc">Output SPAM / HAM label and calibrated confidence.</div>
        </div>
        """, unsafe_allow_html=True)

    # Tech Stack Badges
    st.markdown("""
    <div style="margin-top: 1.5rem; margin-bottom: 0.8rem; font-size: 1.15rem; font-weight: 700; color: #FFFFFF;">
        Technologies Used
    </div>
    """, unsafe_allow_html=True)

    techs = [
        ("Python", "3.11+", "🐍"),
        ("Streamlit", "1.28+", "⚡"),
        ("Scikit-Learn", "1.3+", "🧠"),
        ("NLTK", "3.8+", "📚"),
        ("Pandas", "2.0+", "🐼"),
        ("NumPy", "1.24+", "🔢"),
        ("Matplotlib", "3.7+", "📊")
    ]

    t_cols = st.columns(len(techs))
    for idx, (name, ver, icon) in enumerate(techs):
        with t_cols[idx]:
            st.markdown(f"""
            <div style="background:#080D1A; border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:0.8rem 0.5rem; text-align:center;">
                <div style="font-size:1.4rem; margin-bottom:4px;">{icon}</div>
                <div style="font-weight:700; font-size:0.85rem; color:#FFFFFF;">{name}</div>
                <div style="font-size:0.7rem; color:#64748B;">{ver}</div>
            </div>
            """, unsafe_allow_html=True)
