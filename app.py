from flask import Flask
app = Flask(__name__) # This is the "Entry Point" Vercel is looking for

@app.route('/')
def home():

import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import string

# --- INITIALIZATION ---
# Professionals download necessary NLTK data at the start
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="IBM VeriFact NLP", layout="wide", initial_sidebar_state="expanded")

# --- STUNNING CSS STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=Playfair+Display:wght@700;800&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        min-height: 100vh;
        color: #1a1a1a;
    }
    
    /* Page background overlay */
    .main {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 50%, rgba(240, 147, 251, 0.95) 100%);
        padding: 2rem !important;
        border-radius: 20px;
    }
    
    /* Title styling - gorgeous and bold */
    .stTitle {
        display: none !important;
    }
    
    .centered-title {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #e0d5ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        letter-spacing: -1px;
        text-align: center;
        width: 100%;
        margin: 0 auto 1.5rem auto;
    }
    
    /* Markdown headers */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border: 3px solid #ffffff !important;
        border-radius: 15px !important;
        background-color: rgba(255, 255, 255, 0.98) !important;
        font-size: 1rem !important;
        padding: 1.5rem !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s ease;
        color: #1a1a1a !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #999 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #ffd700 !important;
        box-shadow: 0 15px 50px rgba(255, 215, 0, 0.4) !important;
    }
    
    /* Button styling - premium look */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 12px 40px !important;
        border: 3px solid white !important;
        border-radius: 50px !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%);
        box-shadow: 0 15px 40px rgba(240, 147, 251, 0.6);
        transform: translateY(-2px);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Metrics styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(240,245,255,0.95) 100%);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
        border-left: 5px solid #667eea;
        margin: 15px 0;
    }
    
    .metric-value {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 10px;
    }
    
    /* Info boxes */
    .success-box {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(102, 187, 106, 0.15) 100%);
        border-left: 5px solid #4CAF50;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        color: #2e7d32;
        font-weight: 600;
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 152, 0, 0.1) 0%, rgba(255, 193, 7, 0.15) 100%);
        border-left: 5px solid #ff9800;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        color: #e65100;
        font-weight: 600;
    }
    
    .danger-box {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.1) 0%, rgba(229, 57, 53, 0.15) 100%);
        border-left: 5px solid #f44336;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        color: #c62828;
        font-weight: 600;
    }
    
    /* Column containers */
    .stColumn {
        background: transparent !important;
        border-radius: 0 !important;
        padding: 0 !important;
        backdrop-filter: none !important;
        box-shadow: none !important;
        border: none !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%) !important;
        border-radius: 10px !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Report findings */
    .finding-item {
        background: rgba(255, 255, 255, 0.95);
        border-left: 4px solid #667eea;
        padding: 15px 20px;
        margin: 10px 0;
        border-radius: 8px;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Smooth transitions */
    * {
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# --- ADVANCED NLP PREPROCESSING FUNCTION ---
def preprocess_nlp(text):
    """Advanced NLP Preprocessing with Lemmatization & Tokenization."""
    # 1. Lowercasing & Whitespace Cleaning
    text = text.lower().strip()
    
    # 2. Sentence tokenization for deeper analysis
    sentences = sent_tokenize(text)
    
    # 3. Word tokenization with punctuation removal
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalpha()]
    
    # 4. Lemmatization for better feature extraction
    lemmatized = [lemmatizer.lemmatize(w) for w in words]
    
    # 5. Stopword removal
    filtered_words = [w for w in lemmatized if w not in stop_words and len(w) > 2]
    
    return " ".join(filtered_words), len(words), len(filtered_words), sentences, lemmatized

# --- ADVANCED SENTIMENT ANALYSIS ---
def analyze_sentiment(text):
    """Analyze sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner)."""
    scores = sia.polarity_scores(text)
    return scores

# --- LINGUISTIC FEATURES EXTRACTION ---
def extract_features(text, cleaned_words):
    """Extract advanced linguistic features for accuracy improvement."""
    features = {}
    
    # 1. Word frequency analysis
    word_freq = Counter(cleaned_words)
    features['avg_word_freq'] = sum(word_freq.values()) / len(word_freq) if word_freq else 0
    features['unique_words'] = len(word_freq)
    features['most_common'] = word_freq.most_common(3)
    
    # 2. Readability metrics
    sentences = sent_tokenize(text)
    features['avg_sentence_length'] = len(text.split()) / len(sentences) if sentences else 0
    features['sentence_count'] = len(sentences)
    
    # 3. Character analysis
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    
    # 4. Emotional intensity
    emotional_words = {
        'shocked', 'amazing', 'incredible', 'terrible', 'horrible', 'awful',
        'fantastic', 'devastating', 'unbelievable', 'stunning', 'outrageous'
    }
    features['emotional_words_count'] = sum(1 for w in cleaned_words if w in emotional_words)
    
    # 5. Bias indicators
    bias_words = {
        'obviously', 'clearly', 'definitely', 'surely', 'undoubtedly', 'certainly',
        'always', 'never', 'all', 'none', 'every', 'any'
    }
    features['bias_words_count'] = sum(1 for w in cleaned_words if w in bias_words)
    
    return features

# --- CORE ANALYSIS LOGIC WITH ADVANCED DETECTION ---
def analyze_text(text):
    """Comprehensive NLP-powered analysis with multiple detection algorithms."""
    cleaned_text, original_len, cleaned_len, sentences, lemmatized = preprocess_nlp(text)
    features = extract_features(text, lemmatized)
    sentiment = analyze_sentiment(text)
    
    reasons = []
    score = 100
    risk_level = "Low Risk"
    
    # ===== RULE 1: SENSATIONALISM & EMOTIONAL MANIPULATION =====
    if features['emotional_words_count'] > cleaned_len * 0.05:
        reasons.append(f"🚩 High emotional intensity detected ({features['emotional_words_count']} intensive words) - likely clickbait")
        score -= 25
    
    # ===== RULE 2: PUNCTUATION ANOMALIES =====
    exclamation_count = text.count('!')
    question_count = text.count('?')
    if text.count('!!') > 0 or text.count('??') > 0:
        reasons.append("⚠️ Excessive punctuation indicating sensationalism")
        score -= 15
    elif exclamation_count > len(sentences) * 0.5:
        reasons.append(f"⚠️ Unusually high exclamation marks ({exclamation_count} in {len(sentences)} sentences)")
        score -= 18
    
    # ===== RULE 3: BIAS & ABSOLUTISM =====
    if features['bias_words_count'] > cleaned_len * 0.08:
        reasons.append(f"📌 Strong opinionated language detected - {features['bias_words_count']} bias indicators")
        score -= 22
    
    # ===== RULE 4: SENTIMENT EXTREMISM =====
    neg_sentiment = sentiment['neg']
    pos_sentiment = sentiment['pos']
    
    if neg_sentiment > 0.7:
        reasons.append(f"⛔ Extremely negative sentiment ({neg_sentiment*100:.0f}%) - manipulation attempt")
        score -= 28
    elif pos_sentiment > 0.7:
        reasons.append(f"✨ Extremely positive sentiment ({pos_sentiment*100:.0f}%) - hype/propaganda")
        score -= 20
    
    # ===== RULE 5: SENTENCE LENGTH ANALYSIS =====
    if features['avg_sentence_length'] < 8:
        reasons.append(f"⚡ Very short sentences (avg: {features['avg_sentence_length']:.1f} words) - sensationalism technique")
        score -= 12
    elif features['avg_sentence_length'] > 30:
        reasons.append(f"📚 Excessively long sentences - potential obfuscation")
        score -= 15
    
    # ===== RULE 6: READABILITY FOR MANIPULATION =====
    if features['uppercase_ratio'] > 0.15:
        reasons.append("🔤 Excessive CAPITALIZATION - aggressive tone")
        score -= 18
    
    # ===== RULE 7: SUSPICIOUS PATTERNS & KEYWORDS =====
    suspicious_keywords = {
        'exclusive', 'revealed', 'shocking', 'secret', 'banned', 'coverup', 'conspiracy',
        'unmasked', 'exposed', 'hidden', 'suppressed', 'leaked', 'scandal', 'bombshell'
    }
    suspicious_count = sum(1 for word in lemmatized if word in suspicious_keywords)
    if suspicious_count > 0:
        reasons.append(f"🔍 {suspicious_count} suspicious keywords detected - potential misinformation")
        score -= (10 * suspicious_count)
    
    # ===== RULE 8: SOURCE & EVIDENCE INDICATORS =====
    evidence_keywords = {'study', 'research', 'scientists', 'prove', 'data', 'evidence', 'survey', 'analysis'}
    evidence_count = sum(1 for word in lemmatized if word in evidence_keywords)
    if evidence_count == 0 and original_len > 100:
        reasons.append("📋 No evidence/research citations detected")
        score -= 15
    else:
        reasons.append(f"✅ {evidence_count} evidence/research keywords found")
        score += 10
    
    # ===== RULE 9: ATTRIBUTE CLAIMS (ATTRIBUTIONS) =====
    attribution_keywords = {'according', 'said', 'reported', 'claim', 'told', 'show', 'reveal'}
    attribution_count = sum(1 for word in lemmatized if word in attribution_keywords)
    if attribution_count == 0 and original_len > 100:
        reasons.append("⚠️ Claims made without proper attribution")
        score -= 12
    
    # ===== RULE 10: TEXT LENGTH CREDIBILITY =====
    if original_len < 50:
        reasons.append("📏 Very short text - insufficient for credibility assessment")
        score -= 10
    elif original_len > 300:
        score += 5
        reasons.append("✅ Substantial content length - more credible")
    
    # ===== DETERMINE RISK LEVEL =====
    if score >= 80:
        risk_level = "✅ Low Risk - Likely Credible"
    elif score >= 60:
        risk_level = "⚠️ Medium Risk - Verify Details"
    else:
        risk_level = "🚩 High Risk - Likely Misinformation"
    
    # Ensure score stays within bounds
    score = max(0, min(100, score))
    
    return score, reasons, original_len, cleaned_len, sentiment, features, risk_level

# --- GORGEOUS INTERFACE ---
st.markdown("""
<style>
    @keyframes glow {
        0%, 100% { text-shadow: 0 0 10px rgba(255, 215, 0, 0.5), 0 0 20px rgba(102, 126, 234, 0.3); }
        50% { text-shadow: 0 0 20px rgba(255, 215, 0, 0.8), 0 0 30px rgba(102, 126, 234, 0.6); }
    }
    
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .hero-title {
        animation: glow 3s ease-in-out infinite;
    }
</style>
<div class="centered-title hero-title">Truth Lens 🔭</div>
<div style="text-align: center; margin-bottom: 2rem; animation: slideDown 1s ease-out;">
    <div style="background: linear-gradient(135deg, rgba(255,215,0,0.15) 0%, rgba(240,147,251,0.15) 50%, rgba(102,126,234,0.15) 100%); padding: 2rem 2.5rem; border-radius: 20px; backdrop-filter: blur(15px); border: 2px solid rgba(255,215,0,0.4); margin: 1rem auto; max-width: 700px; box-shadow: 0 0 30px rgba(255,215,0,0.2);">
        <p style="font-family: 'Playfair Display', serif; font-size: 1.6rem; font-weight: 700; background: linear-gradient(135deg, #ffd700 0%, #ffffff 50%, #ffd700 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0.5rem 0; letter-spacing: 1px;">✨ Advanced NLP & Machine Learning ✨</p>
        <div style="height: 3px; background: linear-gradient(90deg, transparent 0%, #ffd700 20%, #ffd700 80%, transparent 100%); margin: 1.2rem 0; width: 70%; margin-left: auto; margin-right: auto; border-radius: 2px; box-shadow: 0 0 15px rgba(255,215,0,0.5);"></div>
        <p style="font-size: 1.15rem; color: #fff; font-weight: 400; margin: 1rem 0; letter-spacing: 1.5px; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">🔬 Intelligent Fact Verification Engine</p>
        <p style="font-size: 0.95rem; color: rgba(255,255,255,0.9); letter-spacing: 3px; margin: 0.8rem 0; text-transform: uppercase; font-weight: 600; font-style: italic;">💼 IBM AI/ML Internship Project</p>
        <p style="font-size: 0.85rem; color: rgba(255,215,0,0.8); margin-top: 1rem; letter-spacing: 1px;">Now analyzing with Deep NLP Technology</p>
    </div>
</div>
""", unsafe_allow_html=True)

user_input = st.text_area("📝 Paste Your News Article Below:", height=200, placeholder="Enter the text you want to analyze...")

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_button = st.button("🚀 Run Advanced Analysis", use_container_width=True)

if analyze_button:
    if user_input.strip():
        score, findings, o_len, c_len, sentiment, features, risk_level = analyze_text(user_input)
        
        st.markdown("---")
        
        # === TOP METRICS ROW ===
        met1, met2, met3, met4 = st.columns(4)
        
        with met1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{score}/100</div>
                <div class="metric-label">Credibility Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with met2:
            risk_color = "✅" if "Low" in risk_level else ("⚠️" if "Medium" in risk_level else "🚩")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size: 1.8rem;">{risk_color}</div>
                <div class="metric-label">{risk_level.split('-')[1].strip() if '-' in risk_level else 'Risk Assessment'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with met3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{o_len}</div>
                <div class="metric-label">Total Words</div>
            </div>
            """, unsafe_allow_html=True)
        
        with met4:
            unique_ratio = (features['unique_words'] / o_len * 100) if o_len > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{unique_ratio:.0f}%</div>
                <div class="metric-label">Vocabulary Diversity</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # === DETAILED ANALYSIS ===
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("🔬 AI Linguistic Report")
            
            if findings:
                for finding in findings:
                    st.markdown(f'<div class="finding-item">{finding}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✨ No red flags detected! Text appears credible.</div>', unsafe_allow_html=True)
        
        with col_right:
            st.subheader("📊 Sentiment Analysis")
            
            sentiment_cols = st.columns(4)
            with sentiment_cols[0]:
                st.metric("Positive", f"{sentiment['pos']*100:.0f}%")
            with sentiment_cols[1]:
                st.metric("Neutral", f"{sentiment['neu']*100:.0f}%")
            with sentiment_cols[2]:
                st.metric("Negative", f"{sentiment['neg']*100:.0f}%")
            with sentiment_cols[3]:
                st.metric("Compound", f"{sentiment['compound']:.2f}")
            
            st.info(f"""
            **Sentiment Interpretation:**
            - Positive: {sentiment['pos']*100:.0f}% (Emotional, promotional)
            - Negative: {sentiment['neg']*100:.0f}% (Critical, alarming)
            - Neutral: {sentiment['neu']*100:.0f}% (Factual, objective)
            """)
        
        st.markdown("---")
        
        # === ADVANCED METRICS ===
        with st.expander("📈 Advanced Linguistic Metrics", expanded=False):
            adv_col1, adv_col2, adv_col3 = st.columns(3)
            
            with adv_col1:
                st.metric("Average Sentence Length", f"{features['avg_sentence_length']:.1f} words")
                st.caption("Lower = More sensational | Higher = Academic")
                st.metric("Emotional Words", features['emotional_words_count'])
                st.metric("Bias Indicators", features['bias_words_count'])
            
            with adv_col2:
                st.metric("Total Sentences", features['sentence_count'])
                st.metric("Unique Words (Diversity)", features['unique_words'])
                st.metric("Capitalization Ratio", f"{features['uppercase_ratio']*100:.1f}%")
            
            with adv_col3:
                st.metric("Filtered Words (NLP)", c_len)
                st.metric("Removed Stopwords", o_len - c_len)
                st.metric("Avg Word Frequency", f"{features['avg_word_freq']:.2f}")
        
        st.markdown("---")
        
        # === FINAL VERDICT ===
        if score >= 80:
            st.markdown(f'<div class="success-box">✅ VERDICT: {risk_level} | This article appears credible. Always cross-reference sources.</div>', unsafe_allow_html=True)
        elif score >= 60:
            st.markdown(f'<div class="warning-box">⚠️ VERDICT: {risk_level} | Verify facts against reliable sources before sharing.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="danger-box">🚩 VERDICT: {risk_level} | High likelihood of misinformation. Do not share.</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# === FOOTER ===
st.markdown("""
---
<div style="text-align: center; color: rgba(255,255,255,0.7); font-size: 0.85rem; margin-top: 3rem;">
    <p>🔬 Advanced NLP Engine | VADER Sentiment Analysis | Machine Learning Feature Extraction</p>
    <p>Disclaimer: This tool assists in credibility assessment. Always verify information from multiple sources.</p>
</div>

""", unsafe_allow_html=True)
