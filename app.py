import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLING
# ============================================================
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# LOAD MODEL AND PREPROCESSING ARTIFACTS
# ============================================================
@st.cache_resource
def load_model_and_artifacts():
    """Load the trained model and character mapping."""
    try:
        model = tf.keras.models.load_model("final_phishing_model.keras")
        
        with open("char2idx.pkl", "rb") as f:
            char2idx = pickle.load(f)
        
        st.write(f"✅ Model loaded successfully")
        st.write(f"✅ char2idx loaded with {len(char2idx)} characters")
        
        return model, char2idx
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found! Error: {e}")
        st.stop()

# ============================================================
# PREPROCESSING FUNCTION
# ============================================================
def encode_url(url, char2idx, maxlen=200):
    """Encode URL to numeric sequence."""
    if not isinstance(url, str):
        return [0] * maxlen
    
    url = url.lower()[:maxlen]
    seq = [char2idx.get(c, 0) for c in url]
    seq += [0] * (maxlen - len(seq))
    return seq

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.title("🔒 Phishing URL Detector")
    st.markdown("Detect phishing URLs using Deep Learning (CNN+LSTM)")
    
    # Load model
    model, char2idx = load_model_and_artifacts()
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info(
            "This app uses a CNN+LSTM neural network trained on 500K+ URLs "
            "to detect phishing websites with 97.7% accuracy."
        )
        st.markdown("---")
        st.subheader("Model Details")
        st.write(f"- **Architecture**: CNN + BiLSTM")
        st.write(f"- **Accuracy**: 97.74%")
        st.write(f"- **Precision**: 95.34%")
        st.write(f"- **Recall**: 96.81%")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Single URL", "📋 Batch Analysis", "📊 Statistics"])
    
    # ============================================================
    # TAB 1: SINGLE URL
    # ============================================================
    with tab1:
        st.subheader("Check a Single URL")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            url_input = st.text_input(
                "Enter URL:",
                placeholder="https://example.com",
                label_visibility="collapsed"
            )
        with col2:
            check_button = st.button("🔍 Check", use_container_width=True)
        
        if check_button and url_input:
            # Encode and predict
            encoded = np.array([encode_url(url_input, char2idx)])
            prediction = model.predict(encoded, verbose=0)[0][0]
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction > 0.5:
                    st.metric("Status", "⚠️ PHISHING", delta="High Risk")
                else:
                    st.metric("Status", "✅ LEGITIMATE", delta="Safe")
            
            with col2:
                st.metric("Confidence", f"{max(prediction, 1-prediction):.1%}")
            
            with col3:
                st.metric("Score", f"{prediction:.4f}")
            
            # Confidence bar
            st.write("**Confidence Breakdown:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Phishing Probability:")
                st.progress(float(prediction))
            with col2:
                st.write("Legitimate Probability:")
                st.progress(float(1 - prediction))
            
            # Details
            st.markdown("---")
            st.subheader("📋 URL Details")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Length**: {len(url_input)}")
            with col2:
                try:
                    tld = url_input.split('.')[-1].split('/')[0]
                    st.write(f"**TLD**: {tld}")
                except:
                    st.write("**TLD**: Unknown")
            with col3:
                st.write(f"**Encoded Length**: 200")
        
        elif check_button:
            st.warning("Please enter a URL!")
    
    # ============================================================
    # TAB 2: BATCH ANALYSIS
    # ============================================================
    with tab2:
        st.subheader("Analyze Multiple URLs")
        
        batch_input = st.text_area(
            "Enter URLs (one per line):",
            placeholder="https://example1.com\nhttps://example2.com",
            height=200
        )
        
        if st.button("📊 Analyze Batch"):
            urls = [url.strip() for url in batch_input.split('\n') if url.strip()]
            
            if not urls:
                st.warning("Please enter at least one URL!")
            else:
                st.info(f"Analyzing {len(urls)} URL(s)...")
                
                # Encode and predict
                encoded_urls = np.array([encode_url(u, char2idx) for u in urls])
                predictions = model.predict(encoded_urls, verbose=0)
                
                # Create results dataframe
                results = []
                for url, pred in zip(urls, predictions):
                    pred_val = pred[0]
                    status = "⚠️ Phishing" if pred_val > 0.5 else "✅ Legitimate"
                    results.append({
                        "URL": url[:50] + "..." if len(url) > 50 else url,
                        "Status": status,
                        "Confidence": f"{max(pred_val, 1-pred_val):.2%}",
                        "Score": f"{pred_val:.4f}"
                    })
                
                # Display table
                st.dataframe(results, use_container_width=True)
                
                # Summary stats
                phishing_count = sum(1 for p in predictions if p[0] > 0.5)
                legit_count = len(urls) - phishing_count
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total URLs", len(urls))
                with col2:
                    st.metric("Phishing", phishing_count, delta=f"{phishing_count/len(urls)*100:.1f}%")
                with col3:
                    st.metric("Legitimate", legit_count, delta=f"{legit_count/len(urls)*100:.1f}%")
    
    # ============================================================
    # TAB 3: MODEL STATS
    # ============================================================
    with tab3:
        st.subheader("Model Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", "97.74%")
        with col2:
            st.metric("AUC Score", "0.9961")
        with col3:
            st.metric("F1 Score", "~0.96")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Precision", "95.34%")
        with col2:
            st.metric("Recall", "96.81%")
        
        st.markdown("---")
        st.subheader("Training Info")
        st.write("""
        - **Dataset Size**: 549,346 URLs
        - **Training Samples**: ~439,477
        - **Test Samples**: ~109,870
        - **Model Architecture**: CNN (128 filters, kernel=5) + BiLSTM (64 units)
        - **Vocabulary Size**: 45 characters
        - **Max URL Length**: 200 characters
        - **Epochs Trained**: 18 (Early stopping)
        """)

if __name__ == "__main__":
    main()
    