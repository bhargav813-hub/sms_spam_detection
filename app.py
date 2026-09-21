"""
SMS Spam Detection System - Streamlit Web Application
A modern, interactive interface for real-time SMS spam classification,
model evaluation, performance comparisons, and exploratory data analysis.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="SMS Spam Detection System",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished aesthetics
st.markdown("""
<style>
    /* Main typography & background hints */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-header {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.08);
        color: #ffffff;
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.3rem;
        color: #ffffff;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #a0aec0;
        margin: 0;
        font-weight: 400;
    }

    .badge-tag {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.8rem;
        margin-right: 0.4rem;
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.4);
    }
    
    .result-card-spam {
        background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%);
        border: 1px solid #ef4444;
        border-radius: 14px;
        padding: 1.5rem;
        color: white;
        margin: 1.2rem 0;
        box-shadow: 0 8px 20px rgba(239, 68, 68, 0.2);
        animation: fadeIn 0.4s ease-in;
    }
    
    .result-card-ham {
        background: linear-gradient(135deg, #052e16 0%, #14532d 100%);
        border: 1px solid #22c55e;
        border-radius: 14px;
        padding: 1.5rem;
        color: white;
        margin: 1.2rem 0;
        box-shadow: 0 8px 20px rgba(34, 197, 94, 0.2);
        animation: fadeIn 0.4s ease-in;
    }

    .metric-box {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #6366f1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .pipeline-step {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #6366f1;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import preprocess_text
from src.train_model import train_and_evaluate

MODELS_DIR = "models"
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "spam_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.pkl")
DATASET_PATH = "data/spam.csv"


@st.cache_resource(show_spinner=False)
def load_model_pipeline():
    """Loads cached models, vectorizer, and evaluation metrics."""
    if not (os.path.exists(CLASSIFIER_PATH) and os.path.exists(VECTORIZER_PATH) and os.path.exists(METRICS_PATH)):
        return None, None, None
    
    try:
        model = joblib.load(CLASSIFIER_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        metrics = joblib.load(METRICS_PATH)
        return model, vectorizer, metrics
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None


# Check and load models
model, vectorizer, metrics_bundle = load_model_pipeline()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/sms.png", width=110)
    st.title("Project Navigator")
    st.markdown("College NLP Project: **SMS Spam Detection System**")
    
    st.markdown("---")
    st.subheader("Model Status")
    if model is not None:
        st.success(f"Active: **{metrics_bundle.get('best_model_name', 'Loaded Model')}**")
        st.caption(f"Vocabulary: {metrics_bundle.get('feature_count', 3000):,} TF-IDF n-grams")
    else:
        st.warning("⚠️ Model files not detected")

    st.markdown("---")
    st.subheader("Quick Actions")
    if st.button("🔄 Retrain Models Now", use_container_width=True):
        with st.spinner("Executing NLP pipeline & training models..."):
            try:
                train_and_evaluate(data_path=DATASET_PATH, output_dir=MODELS_DIR)
                st.cache_resource.clear()
                st.success("Models retrained and saved successfully!")
                st.rerun()
            except Exception as ex:
                st.error(f"Retraining failed: {ex}")

    st.markdown("---")
    st.markdown("""
    **Developer Guide**:
    - **Dataset**: SMS Spam Collection
    - **Techniques**: NLTK + TF-IDF
    - **Models**: Naive Bayes & Logistic Reg.
    - **Metric**: F1-Score & Accuracy
    """)

# Hero Banner
st.markdown("""
<div class="hero-header">
    <div class="hero-title">📱 SMS Spam Detection System</div>
    <div class="hero-subtitle">Intelligent Text Classification using Natural Language Processing & Machine Learning</div>
    <div>
        <span class="badge-tag">NLTK Preprocessing</span>
        <span class="badge-tag">TF-IDF Vectorization</span>
        <span class="badge-tag">Multinomial Naive Bayes</span>
        <span class="badge-tag">Logistic Regression</span>
        <span class="badge-tag">Streamlit UI</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Defensive check: if models are missing
if model is None or vectorizer is None or metrics_bundle is None:
    st.error("### ⚠️ Trained Model Files Missing")
    st.info(
        "The model artifacts (`spam_classifier.pkl`, `tfidf_vectorizer.pkl`, `model_metrics.pkl`) "
        "have not been trained yet. You can train them directly right now."
    )
    if st.button("🚀 Train Models Now (One Click)", type="primary"):
        with st.spinner("Processing dataset, running NLP pipeline, and training models..."):
            try:
                train_and_evaluate(data_path=DATASET_PATH, output_dir=MODELS_DIR)
                st.cache_resource.clear()
                st.success("Models trained successfully! Refreshing app...")
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")
    st.stop()

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 SMS Prediction",
    "📊 Model Performance",
    "📈 Exploratory Data Analysis",
    "ℹ️ Project Architecture"
])

# -------------------------------------------------------------
# TAB 1: SMS PREDICTION
# -------------------------------------------------------------
with tab1:
    st.markdown("### Real-time SMS Classifier")
    st.write("Type or paste an SMS message below, or click any sample message to run live NLP inference.")

    # Preset examples
    sample_messages = [
        "Congratulations! You have won a £1,000 cash prize or a £2,000 holiday! Claim now. Call 09050000327 to claim.",
        "Hey bro, are we meeting at the college library at 4pm for group study?",
        "URGENT: Your account has been suspended due to suspicious activity. Verify now at http://secure-bank-login.com to unlock.",
        "Can you please pick up some groceries on your way back home? Thanks!",
        "WINNER! As a valued customer you have been selected to receive a free £900 gift prize. Reply CLAIM to 87575 now.",
        "I will be reaching the seminar hall in 10 minutes. Please save a seat for me."
    ]

    st.markdown("**Test with Example Messages:**")
    cols = st.columns(3)
    for idx, sample in enumerate(sample_messages):
        btn_label = f"📝 Sample {idx+1}: " + (sample[:38] + "..." if len(sample) > 38 else sample)
        if cols[idx % 3].button(btn_label, key=f"example_{idx}", use_container_width=True):
            st.session_state["input_sms"] = sample

    # User input text area
    user_sms = st.text_area(
        "Enter your SMS message:",
        value=st.session_state.get("input_sms", ""),
        height=130,
        placeholder="e.g. Free entry in 2 a weekly competition to win cash prizes! Text WIN to 80085..."
    )

    check_col, clear_col = st.columns([1, 5])
    with check_col:
        check_button = st.button("🔍 Check Message", type="primary", use_container_width=True)
    with clear_col:
        if st.button("🗑️ Clear", use_container_width=False):
            st.session_state["input_sms"] = ""
            st.rerun()

    if check_button or user_sms.strip():
        if not user_sms.strip():
            st.warning("⚠️ Please enter an SMS message before clicking Check Message.")
        else:
            # 1. NLP Preprocessing
            cleaned_sms = preprocess_text(user_sms)
            
            if not cleaned_sms.strip():
                st.warning("⚠️ The message does not contain valid alphanumeric text after cleaning punctuation and stopwords.")
            else:
                # 2. TF-IDF Transformation
                features = vectorizer.transform([cleaned_sms])

                # 3. Model Prediction
                prediction = model.predict(features)[0]
                is_spam = bool(prediction == 1)

                # 4. Probability / Confidence Calculation
                proba_spam = 0.5
                proba_ham = 0.5
                confidence = 50.0

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(features)[0]
                    proba_ham = float(probs[0])
                    proba_spam = float(probs[1])
                    confidence = proba_spam * 100 if is_spam else proba_ham * 100

                # Display Result Box
                if is_spam:
                    st.markdown(f"""
                    <div class="result-card-spam">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size: 1.6rem; font-weight:800; letter-spacing:-0.5px;">🚨 SPAM MESSAGE</span>
                            <span style="background:rgba(255,255,255,0.2); padding:4px 12px; border-radius:20px; font-weight:700;">
                                {confidence:.2f}% Confidence
                            </span>
                        </div>
                        <p style="margin-top:0.6rem; margin-bottom:0; font-size:0.95rem; opacity:0.9;">
                            This message contains unsolicited promotional patterns, urgent reward language, or suspicious spam characteristics.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card-ham">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size: 1.6rem; font-weight:800; letter-spacing:-0.5px;">✅ HAM — NOT SPAM</span>
                            <span style="background:rgba(255,255,255,0.2); padding:4px 12px; border-radius:20px; font-weight:700;">
                                {confidence:.2f}% Confidence
                            </span>
                        </div>
                        <p style="margin-top:0.6rem; margin-bottom:0; font-size:0.95rem; opacity:0.9;">
                            This message appears to be legitimate, natural communication (Ham).
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Probabilities breakdown
                st.markdown("#### Prediction Confidence Breakdown")
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    st.write(f"**Ham (Legitimate) Probability**: `{proba_ham * 100:.2f}%`")
                    st.progress(float(proba_ham))
                with p_col2:
                    st.write(f"**Spam Probability**: `{proba_spam * 100:.2f}%`")
                    st.progress(float(proba_spam))

                # Step-by-step NLP Preprocessing inspection
                with st.expander("🔍 Inspect NLP Preprocessing Transformation", expanded=False):
                    st.markdown("**Original Raw Message:**")
                    st.code(user_sms, language="text")

                    st.markdown("**Preprocessed Tokens (Lowercased, Punctuation & Stopwords Removed, Porter Stemmed):**")
                    st.code(cleaned_sms, language="text")

                    st.markdown("**Identified Active TF-IDF Vocabulary Features:**")
                    # Extract non-zero TF-IDF features
                    feature_names = np.array(vectorizer.get_feature_names_out())
                    non_zero_indices = features.indices
                    if len(non_zero_indices) > 0:
                        feature_scores = features.data
                        matched_df = pd.DataFrame({
                            "Stemmed Feature": feature_names[non_zero_indices],
                            "TF-IDF Weight": np.round(feature_scores, 4)
                        }).sort_values(by="TF-IDF Weight", ascending=False)
                        st.dataframe(matched_df.head(10), use_container_width=True, hide_index=True)
                    else:
                        st.info("No matching vocabulary features found in the trained TF-IDF vocabulary.")


# -------------------------------------------------------------
# TAB 2: MODEL PERFORMANCE & EVALUATION
# -------------------------------------------------------------
with tab2:
    st.markdown("### Model Evaluation & Comparative Performance")
    st.write(
        f"The system trained and evaluated multiple candidate algorithms on an **80/20 stratified split** "
        f"({int(metrics_bundle.get('eda_stats', {}).get('cleaned_rows', 5169) * 0.8)} train / "
        f"{int(metrics_bundle.get('eda_stats', {}).get('cleaned_rows', 5169) * 0.2)} test). "
        f"The best model was chosen dynamically using **F1-score**."
    )

    best_name = metrics_bundle.get("best_model_name", "Multinomial Naive Bayes")
    best_m = metrics_bundle.get("best_metrics", {})

    # Metric KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{best_m.get('accuracy', 0)*100:.2f}%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{best_m.get('precision', 0)*100:.2f}%</div>
            <div class="metric-label">Precision (Spam)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{best_m.get('recall', 0)*100:.2f}%</div>
            <div class="metric-label">Recall (Spam)</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{best_m.get('f1_score', 0)*100:.2f}%</div>
            <div class="metric-label">F1-Score (Primary Metric)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Comparison Table and Confusion Matrix
    col_comp, col_cm = st.columns([1.1, 1])

    with col_comp:
        st.markdown("#### 🏆 Model Comparison Table")
        comp_records = metrics_bundle.get("comparison_table", [])
        if comp_records:
            comp_df = pd.DataFrame(comp_records)
            st.dataframe(
                comp_df.style.highlight_max(
                    subset=["Accuracy", "Precision", "Recall", "F1-Score"],
                    color="#312e81"
                ),
                use_container_width=True,
                hide_index=True
            )
        
        st.markdown(f"""
        > **Dynamic Selection Outcome**: **{best_name}** achieved the highest F1-Score 
        > ({best_m.get('f1_score', 0)*100:.2f}%), offering superior precision and minimizing false spam alerts.
        """)

        st.markdown("#### Detailed Classification Report")
        report = best_m.get("classification_report", {})
        if report:
            rows = []
            for class_name in ["Ham", "Spam"]:
                if class_name in report:
                    rows.append({
                        "Class": class_name,
                        "Precision": f"{report[class_name]['precision']*100:.2f}%",
                        "Recall": f"{report[class_name]['recall']*100:.2f}%",
                        "F1-Score": f"{report[class_name]['f1-score']*100:.2f}%",
                        "Support": int(report[class_name]['support'])
                    })
            st.table(pd.DataFrame(rows))

    with col_cm:
        st.markdown(f"#### 🎯 Confusion Matrix: {best_name}")
        cm = best_m.get("confusion_matrix", [[0, 0], [0, 0]])
        
        fig, ax = plt.subplots(figsize=(4.8, 3.8), dpi=120)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted Ham", "Predicted Spam"],
            yticklabels=["Actual Ham", "Actual Spam"],
            cbar=False,
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"}
        )
        ax.set_title(f"Test Set Evaluation ({sum(map(sum, cm))} SMS)", fontsize=11, fontweight="bold", pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
        st.caption(
            f"**True Negatives (Ham):** {tn} | **False Positives (Ham as Spam):** {fp} | "
            f"**False Negatives (Missed Spam):** {fn} | **True Positives (Spam):** {tp}"
        )


# -------------------------------------------------------------
# TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
# -------------------------------------------------------------
with tab3:
    st.markdown("### Exploratory Data Analysis on Real Dataset")
    eda = metrics_bundle.get("eda_stats", {})

    # Overview KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Messages (Raw)", f"{eda.get('initial_rows', 5572):,}")
    with k2:
        st.metric("Duplicates Removed", f"{eda.get('duplicates_removed', 403):,}")
    with k3:
        st.metric("Unique Cleaned Messages", f"{eda.get('cleaned_rows', 5169):,}")
    with k4:
        st.metric("Spam Ratio", f"{eda.get('class_percentages', {}).get('spam', 12.63)}%")

    st.markdown("---")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### Class Distribution (Ham vs. Spam)")
        dist = eda.get("class_distribution", {"ham": 4516, "spam": 653})
        
        fig_pie, ax_pie = plt.subplots(figsize=(4.5, 3.8), dpi=120)
        colors = ["#22c55e", "#ef4444"]
        labels = [f"Ham ({dist.get('ham', 0)})", f"Spam ({dist.get('spam', 0)})"]
        wedges, texts, autotexts = ax_pie.pie(
            [dist.get("ham", 0), dist.get("spam", 0)],
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            textprops={"fontsize": 10}
        )
        for at in autotexts:
            at.set_color("white")
            at.set_weight("bold")
        ax_pie.axis("equal")
        plt.tight_layout()
        st.pyplot(fig_pie)
        plt.close(fig_pie)

    with col_chart2:
        st.markdown("#### Message Length Comparison (Characters)")
        ham_len = eda.get("ham_char_mean", 70.4)
        spam_len = eda.get("spam_char_mean", 138.8)
        
        fig_bar, ax_bar = plt.subplots(figsize=(5.0, 3.8), dpi=120)
        categories = ["Ham (Legitimate)", "Spam"]
        lengths = [ham_len, spam_len]
        bar_colors = ["#3b82f6", "#f43f5e"]
        bars = ax_bar.bar(categories, lengths, color=bar_colors, width=0.55)
        ax_bar.set_ylabel("Average Character Count", fontsize=10)
        ax_bar.set_title("Spam Messages are ~2x Longer on Average", fontsize=11, fontweight="bold")
        for bar in bars:
            yval = bar.get_height()
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 2,
                f"{yval:.1f} chars",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10
            )
        ax_bar.set_ylim(0, max(lengths) * 1.2)
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    st.markdown("---")

    # Top Words Analysis
    st.markdown("#### Most Frequent Words (Post-NLP Preprocessing)")
    col_w1, col_w2 = st.columns(2)

    top_spam = eda.get("top_spam_words", [])[:10]
    top_ham = eda.get("top_ham_words", [])[:10]

    with col_w1:
        st.markdown("**Top 10 Stemmed Words in Spam**")
        if top_spam:
            s_words, s_counts = zip(*top_spam)
            fig_s, ax_s = plt.subplots(figsize=(5.5, 3.8), dpi=120)
            ax_s.barh(list(reversed(s_words)), list(reversed(s_counts)), color="#ef4444")
            ax_s.set_xlabel("Frequency")
            plt.tight_layout()
            st.pyplot(fig_s)
            plt.close(fig_s)

    with col_w2:
        st.markdown("**Top 10 Stemmed Words in Ham**")
        if top_ham:
            h_words, h_counts = zip(*top_ham)
            fig_h, ax_h = plt.subplots(figsize=(5.5, 3.8), dpi=120)
            ax_h.barh(list(reversed(h_words)), list(reversed(h_counts)), color="#22c55e")
            ax_h.set_xlabel("Frequency")
            plt.tight_layout()
            st.pyplot(fig_h)
            plt.close(fig_h)


# -------------------------------------------------------------
# TAB 4: PROJECT ARCHITECTURE & DETAILS
# -------------------------------------------------------------
with tab4:
    st.markdown("### System Architecture & Technical Methodology")
    
    st.markdown("""
    #### 1. End-to-End Pipeline
    ```
    Raw SMS Dataset (data/spam.csv)
          ↓
    Data Cleaning & Deduplication (5,572 -> 5,169 unique)
          ↓
    NLP Preprocessing (NLTK Tokenization, Stopwords, Porter Stemming)
          ↓
    80/20 Stratified Train-Test Split (random_state=42)
          ↓
    TF-IDF Vectorization (fit on Train, transform on Test)
          ↓
    Model Training (Multinomial Naive Bayes & Logistic Regression)
          ↓
    Dynamic Evaluation & F1-Score Selection
          ↓
    Serialized Storage (joblib .pkl artifacts)
          ↓
    Interactive Streamlit UI Inference
    ```
    """)

    st.markdown("#### 2. Applied NLP Techniques")
    st.markdown("""
    - **Lowercase Conversion**: Normalizes character casing across all words (`Free` -> `free`).
    - **Punctuation & Special Character Removal**: Removes non-alphanumeric noise, URLs, and phone format markers.
    - **Tokenization (`nltk.word_tokenize`)**: Segments sentences into individual word tokens.
    - **Stopword Removal (`nltk.corpus.stopwords`)**: Filters high-frequency, non-discriminative grammatical words (`the`, `is`, `at`).
    - **Porter Stemmer (`nltk.stem.PorterStemmer`)**: Reduces morphological variants to root stems (`winning`, `winner`, `won` -> `win`).
    """)

    st.markdown("#### 3. TF-IDF & Machine Learning Models")
    st.markdown("""
    - **TF-IDF (Term Frequency - Inverse Document Frequency)**:
      Weights words according to their distinctiveness across the corpus, penalizing words that appear everywhere.
      `ngram_range=(1, 2)` captures both single words and pairs of words (e.g. *free entry*, *cash prize*).
    - **Multinomial Naive Bayes (`MultinomialNB`)**:
      A probabilistic classifier based on Bayes' theorem, assuming independent feature distributions. Well-suited for sparse text data.
    - **Logistic Regression (`LogisticRegression`)**:
      A linear model utilizing the sigmoid function to estimate class membership probabilities with balanced weighting.
    """)

    st.markdown("#### 4. Model Selection Rationale")
    st.markdown("""
    Spam detection exhibits natural class imbalance (~87% Ham, ~13% Spam). In this domain, Accuracy alone can be misleading.
    Therefore, **F1-Score** (harmonic mean of Precision and Recall) serves as the primary selection criterion to ensure both
    high spam detection recall and minimal false spam alarms.
    """)
