import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import time

# Configuration de la page
st.set_page_config(
    page_title="AI Agent - Ticket Classification",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé professionnel
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 4px solid #667eea;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, rgb(38, 39, 48) 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 4px solid #ffc107;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s;
    }
    
    /* Custom metrics */
    .custom-metric {
        text-align: center;
        padding: 1rem;
    }
    
    .custom-metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .custom-metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    /* Confidence badge */
    .confidence-high {
        color: #28a745;
        font-weight: 700;
    }
    
    .confidence-medium {
        color: #ffc107;
        font-weight: 700;
    }
    
    .confidence-low {
        color: #dc3545;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# URL du service API
API_URL = "http://127.0.0.1:8000"

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    custom_api_url = st.text_input(
        "API Endpoint", 
        value=API_URL,
        help="URL du service Agent IA"
    )
    if custom_api_url:
        API_URL = custom_api_url
    
    st.markdown("---")
    
    # Health check avec style amélioré
    st.markdown("### 🏥 Service Status")
    status_placeholder = st.empty()
    
    try:
        health = requests.get(f"{API_URL}/health", timeout=3)
        if health.status_code == 200:
            status_placeholder.markdown(
                '<div class="status-badge status-success">✓ ONLINE</div>', 
                unsafe_allow_html=True
            )
        else:
            status_placeholder.markdown(
                '<div class="status-badge status-error">✗ ERROR</div>', 
                unsafe_allow_html=True
            )
    except:
        status_placeholder.markdown(
            '<div class="status-badge status-error">✗ OFFLINE</div>', 
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Model information
    st.markdown("### 🤖 Models")
    with st.expander("TF-IDF Model", expanded=False):
        st.markdown("""
        **Best for:**
        - Short texts (≤20 words)
        - Simple queries
        - English language
        - Fast processing
        """)
    
    with st.expander("Transformer Model", expanded=False):
        st.markdown("""
        **Best for:**
        - Long texts (>20 words)
        - Complex queries
        - Multilingual support
        - Nuanced understanding
        """)
    
    st.markdown("---")
    
    # History
    st.markdown("### 📜 Recent Classifications")
    if st.session_state.history:
        for idx, item in enumerate(reversed(st.session_state.history[-5:])):
            with st.container():
                st.markdown(f"**{item['category']}** ({item['confidence']:.0%})")
                st.caption(f"{item['text'][:40]}..." if len(item['text']) > 40 else item['text'])
                st.markdown("---")
    else:
        st.caption("No classifications yet")
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# Main content
st.markdown('<div class="main-header">🤖 AI Ticket Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Intelligent routing system powered by machine learning</div>', unsafe_allow_html=True)

# Quick stats at the top
if st.session_state.history:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="custom-metric">
            <div class="custom-metric-label">Total</div>
            <div class="custom-metric-value">{len(st.session_state.history)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_conf = sum(h['confidence'] for h in st.session_state.history) / len(st.session_state.history)
        st.markdown(f"""
        <div class="custom-metric">
            <div class="custom-metric-label">Avg Confidence</div>
            <div class="custom-metric-value">{avg_conf:.0%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        tfidf_count = sum(1 for h in st.session_state.history if h['model'] == 'tfidf')
        st.markdown(f"""
        <div class="custom-metric">
            <div class="custom-metric-label">TF-IDF</div>
            <div class="custom-metric-value">{tfidf_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        transformer_count = sum(1 for h in st.session_state.history if h['model'] == 'transformer')
        st.markdown(f"""
        <div class="custom-metric">
            <div class="custom-metric-label">Transformer</div>
            <div class="custom-metric-value">{transformer_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

# Input section with better layout
st.markdown('<div class="section-header">📝 Enter Ticket</div>', unsafe_allow_html=True)

# Example selector at the top
example_col1, example_col2 = st.columns([3, 1])

with example_col1:
    examples = st.selectbox(
        "Quick Examples",
        [
            "Type your own text...",
            "I broke my pc",
            "j'ai un problem d'acces",
            "Cannot access Outlook account from company laptop",
            "Le système plante lors du téléchargement de gros fichiers",
            "Need admin rights to install new software",
            "Printer on 3rd floor is jammed again"
        ],
        label_visibility="collapsed"
    )

with example_col2:
    if st.button("🔄 Load Example", use_container_width=True):
        if examples != "Type your own text...":
            st.session_state.example_text = examples

# Text input
ticket_text = st.text_area(
    "Ticket Description",
    height=120,
    placeholder="Enter the ticket description here... (e.g., 'Cannot login to my email account')",
    value=st.session_state.get('example_text', ''),
    label_visibility="collapsed"
)

# Clear example text after using it
if 'example_text' in st.session_state:
    del st.session_state.example_text

# Action buttons
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    predict_button = st.button("🔮 Classify Ticket", type="primary", use_container_width=True)

with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)
    if clear_button:
        st.rerun()

with col3:
    word_count = len(ticket_text.split()) if ticket_text else 0
    st.metric("Words", word_count)

# Prediction logic
if predict_button and ticket_text:
    start_time = time.time()
    
    with st.spinner("🔄 Analyzing ticket..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"text": ticket_text},
                timeout=30
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Add to history
                st.session_state.history.append({
                    'text': ticket_text,
                    'category': result['category'],
                    'confidence': result['confidence'],
                    'model': result['model_used']
                })
                
                # Success message
                st.markdown(f"""
                <div class="success-box fade-in">
                    <strong>✅ Classification Complete</strong> - Processed in {processing_time:.2f}s
                </div>
                """, unsafe_allow_html=True)
                
                # Main results
                st.markdown('<div class="section-header">📊 Results</div>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("**Category**")
                    st.markdown(f"### {result['category']}")
                
                with col2:
                    confidence_pct = result['confidence'] * 100
                    confidence_class = "confidence-high" if confidence_pct >= 70 else "confidence-medium" if confidence_pct >= 50 else "confidence-low"
                    st.markdown("**Confidence**")
                    st.markdown(f"### <span class='{confidence_class}'>{confidence_pct:.1f}%</span>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("**Model Used**")
                    model_badge_color = "#667eea" if result['model_used'] == 'tfidf' else "#764ba2"
                    st.markdown(f"### <span style='color: {model_badge_color}'>{result['model_used'].upper()}</span>", unsafe_allow_html=True)
                
                with col4:
                    st.markdown("**Text Length**")
                    st.markdown(f"### {word_count} words")
                
                # Model explanation
                st.markdown(f"""
                <div class="info-box">
                    <strong>💡 Model Selection:</strong> {result['explanation']}
                </div>
                """, unsafe_allow_html=True)
                
                # PII warning if needed
                if result['scrubbed_text'] != ticket_text:
                    st.markdown(f"""
                    <div class="warning-box">
                        <strong>🔒 Privacy Protection:</strong> Sensitive information has been masked<br>
                        <small><code>{result['scrubbed_text']}</code></small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Visualizations
                st.markdown('<div class="section-header">📈 Probability Distribution</div>', unsafe_allow_html=True)
                
                model_output = result['model_output']
                
                # Get probabilities
                if 'probabilities' in model_output:
                    probs_data = model_output['probabilities']
                    labels = [p['label'] for p in probs_data]
                    probabilities = [p['probability'] for p in probs_data]
                elif 'all_scores' in model_output:
                    labels = list(model_output['all_scores'].keys())
                    probabilities = list(model_output['all_scores'].values())
                else:
                    labels = [result['category']]
                    probabilities = [result['confidence']]
                
                # Sort by probability
                sorted_data = sorted(zip(labels, probabilities), key=lambda x: x[1], reverse=True)
                labels, probabilities = zip(*sorted_data)
                
                # Create tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "🥧 Pie Chart", "📉 Detailed Scores"])
                
                with tab1:
                    fig_bar = go.Figure(data=[
                        go.Bar(
                            x=list(probabilities),
                            y=list(labels),
                            orientation='h',
                            marker=dict(
                                color=list(probabilities),
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Probability", thickness=15)
                            ),
                            text=[f"{p:.1%}" for p in probabilities],
                            textposition='auto',
                            hovertemplate='<b>%{y}</b><br>Probability: %{x:.2%}<extra></extra>'
                        )
                    ])
                    
                    fig_bar.update_layout(
                        title="",
                        xaxis_title="Probability",
                        yaxis_title="",
                        height=400,
                        showlegend=False,
                        xaxis=dict(tickformat=".0%"),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif")
                    )
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with tab2:
                    top_n = min(6, len(labels))
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=list(labels[:top_n]),
                        values=list(probabilities[:top_n]),
                        hole=0.4,
                        textinfo='label+percent',
                        marker=dict(
                            colors=px.colors.qualitative.Set3,
                            line=dict(color='white', width=2)
                        ),
                        hovertemplate='<b>%{label}</b><br>Probability: %{value:.2%}<extra></extra>'
                    )])
                    
                    fig_pie.update_layout(
                        title=f"Top {top_n} Categories",
                        height=450,
                        font=dict(family="Inter, sans-serif")
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with tab3:
                    # Detailed table
                    st.markdown("### All Category Scores")
                    for label, prob in zip(labels, probabilities):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.text(label)
                        with col2:
                            st.progress(prob, text=f"{prob:.2%}")
                
                # Confidence gauge
                st.markdown('<div class="section-header">🎯 Confidence Score</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result['confidence'] * 100,
                        title={'text': "Confidence Level", 'font': {'size': 20, 'family': 'Inter'}},
                        number={'suffix': "%", 'font': {'size': 48}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 2},
                            'bar': {'color': "#667eea", 'thickness': 0.75},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 50], 'color': '#fee'},
                                {'range': [50, 75], 'color': '#ffc'},
                                {'range': [75, 100], 'color': '#cfc'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 3},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    
                    fig_gauge.update_layout(
                        height=350,
                        font=dict(family="Inter, sans-serif")
                    )
                    
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col2:
                    st.markdown("### Interpretation")
                    
                    conf_val = result['confidence'] * 100
                    if conf_val >= 75:
                        st.success("**High Confidence** ✓")
                        st.markdown("The model is very confident about this classification.")
                    elif conf_val >= 50:
                        st.warning("**Moderate Confidence** ⚠️")
                        st.markdown("The model has reasonable confidence, but consider reviewing.")
                    else:
                        st.error("**Low Confidence** ✗")
                        st.markdown("The model is uncertain. Manual review recommended.")
                    
                    st.markdown("---")
                    st.metric("Processing Time", f"{processing_time:.3f}s")
                    
                    if 'processing_time' in model_output:
                        st.metric("Model Time", f"{model_output['processing_time']:.3f}s")
                
                # Raw data expander
                with st.expander("🔍 View Raw JSON Response"):
                    st.json(result)
                
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timeout. The service is taking too long to respond.")
        except requests.exceptions.ConnectionError:
            st.error(f"🔌 Connection error. Cannot reach API at {API_URL}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

elif predict_button:
    st.warning("⚠️ Please enter a ticket description before classifying.")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 2rem 0;">
    <strong>AI Ticket Classification System</strong> | Version 1.0<br>
    <small>Powered by TF-IDF & Transformer Models</small>
</div>
""", unsafe_allow_html=True)