import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="NextGen Investors Beta",
    page_icon="📈",
    layout="wide"
)

# App title and disclaimer
st.title("📈 NextGen Investors")
st.write("💼 Get stock recommendations based on fundamental analysis and correlation matrix.")
st.markdown(
    """
    <div style="background-color: #45b7d1; color: white; padding: 12px; border-radius: 25px; font-weight: 600; font-size: 1rem; margin: 1rem 0; text-align: center;">
        ⚠ Disclaimer: This platform offers stock guidance based on fundamental analysis and machine learning but does not constitute financial advice; invest responsibly.
    </div>
    """, 
    unsafe_allow_html=True
)

# Correlation matrix for financial metrics
CORRELATION_MATRIX = {
    "P/E Ratio": {"P/E Ratio": 1.0, "P/B Ratio": 0.0045, "D/E Ratio": -0.0035, "ROE": -0.0178, "ROA": -0.0105},
    "P/B Ratio": {"P/E Ratio": 0.0045, "P/B Ratio": 1.0, "D/E Ratio": 0.8175, "ROE": 0.0281, "ROA": 0.0153},
    "D/E Ratio": {"P/E Ratio": -0.0035, "P/B Ratio": 0.8175, "D/E Ratio": 1.0, "ROE": 0.0678, "ROA": -0.0069},
    "ROE": {"P/E Ratio": -0.0178, "P/B Ratio": 0.0281, "D/E Ratio": 0.0678, "ROE": 1.0, "ROA": 0.0771},
    "ROA": {"P/E Ratio": -0.0105, "P/B Ratio": 0.0153, "D/E Ratio": -0.0069, "ROE": 0.0771, "ROA": 1.0},
}

# Test data for companies and their ratios
@st.cache_data
def load_test_data():
    test_companies = {
        'Apple Inc.': {'P/E Ratio': 28.5, 'P/B Ratio': 12.3, 'D/E Ratio': 1.8, 'ROE': 45.2, 'ROA': 18.7},
        'Microsoft Corp.': {'P/E Ratio': 32.1, 'P/B Ratio': 8.9, 'D/E Ratio': 0.6, 'ROE': 38.4, 'ROA': 16.2},
        'Google (Alphabet)': {'P/E Ratio': 24.8, 'P/B Ratio': 4.2, 'D/E Ratio': 0.2, 'ROE': 26.8, 'ROA': 15.3},
        'Amazon.com Inc.': {'P/E Ratio': 58.2, 'P/B Ratio': 7.1, 'D/E Ratio': 1.1, 'ROE': 22.1, 'ROA': 6.8},
        'Tesla Inc.': {'P/E Ratio': 65.4, 'P/B Ratio': 12.8, 'D/E Ratio': 0.4, 'ROE': 19.3, 'ROA': 9.1},
        'Meta Platforms': {'P/E Ratio': 23.2, 'P/B Ratio': 4.8, 'D/E Ratio': 0.1, 'ROE': 24.6, 'ROA': 17.9},
        'NVIDIA Corp.': {'P/E Ratio': 72.3, 'P/B Ratio': 14.5, 'D/E Ratio': 0.3, 'ROE': 35.8, 'ROA': 22.1},
        'JPMorgan Chase': {'P/E Ratio': 12.4, 'P/B Ratio': 1.6, 'D/E Ratio': 2.8, 'ROE': 15.2, 'ROA': 1.3},
        'Johnson & Johnson': {'P/E Ratio': 16.8, 'P/B Ratio': 3.2, 'D/E Ratio': 0.8, 'ROE': 22.4, 'ROA': 9.8},
        'Coca-Cola Co.': {'P/E Ratio': 26.1, 'P/B Ratio': 9.7, 'D/E Ratio': 1.9, 'ROE': 42.8, 'ROA': 8.9},
        'Intel Corp.': {'P/E Ratio': 18.5, 'P/B Ratio': 2.1, 'D/E Ratio': 0.5, 'ROE': 19.6, 'ROA': 7.4},
        'IBM Corp.': {'P/E Ratio': 22.7, 'P/B Ratio': 6.8, 'D/E Ratio': 2.1, 'ROE': 31.2, 'ROA': 3.8},
        'Oracle Corp.': {'P/E Ratio': 29.4, 'P/B Ratio': 8.3, 'D/E Ratio': 2.3, 'ROE': 38.9, 'ROA': 8.6},
        'Walmart Inc.': {'P/E Ratio': 25.8, 'P/B Ratio': 4.9, 'D/E Ratio': 0.6, 'ROE': 21.3, 'ROA': 6.2},
        'Disney Co.': {'P/E Ratio': 35.2, 'P/B Ratio': 2.8, 'D/E Ratio': 0.9, 'ROE': 8.4, 'ROA': 3.1}
    }
    return test_companies

# FAQ data
@st.cache_data
def load_faq_data():
    faqs = {
        'What is P/E Ratio?': 'P/E Ratio (Price-to-Earnings) measures the current share price relative to earnings per share. Lower P/E may indicate undervalued stock.',
        'What is P/B Ratio?': 'P/B Ratio (Price-to-Book) compares market value to book value. A ratio below 1 might indicate undervaluation.',
        'What is D/E Ratio?': 'D/E Ratio (Debt-to-Equity) shows company leverage. Lower ratios generally indicate less financial risk.',
        'What is ROE?': 'ROE (Return on Equity) measures profitability relative to shareholder equity. Higher ROE indicates efficient use of equity.',
        'What is ROA?': 'ROA (Return on Assets) shows how efficiently a company uses its assets to generate profit.',
        'How does the correlation matrix help?': 'The correlation matrix shows relationships between financial metrics, helping identify patterns and anomalies.',
        'How does the prediction work?': 'Our model uses machine learning algorithms and correlation analysis trained on historical financial data.',
        'Is this financial advice?': 'No, this is an educational tool. Always consult with financial advisors before making investment decisions.',
        'What factors affect stock prices?': 'Stock prices are influenced by company performance, market conditions, economic factors, and investor sentiment.'
    }
    return faqs

def calculate_correlation_score(metrics):
    """Calculate a composite score based on correlation matrix"""
    metric_names = ['P/E Ratio', 'P/B Ratio', 'D/E Ratio', 'ROE', 'ROA']
    values = [metrics[name] for name in metric_names]
    
    # Normalize values to 0-1 scale for scoring
    normalized = {}
    normalized['P/E Ratio'] = max(0, min(1, (40 - metrics['P/E Ratio']) / 40))  # Lower P/E is better
    normalized['P/B Ratio'] = max(0, min(1, (10 - metrics['P/B Ratio']) / 10))  # Lower P/B is better
    normalized['D/E Ratio'] = max(0, min(1, (3 - metrics['D/E Ratio']) / 3))    # Lower D/E is better
    normalized['ROE'] = max(0, min(1, metrics['ROE'] / 50))                     # Higher ROE is better
    normalized['ROA'] = max(0, min(1, metrics['ROA'] / 25))                     # Higher ROA is better
    
    # Calculate weighted score using correlation strengths
    correlation_score = 0
    total_weight = 0
    
    for metric1 in metric_names:
        for metric2 in metric_names:
            if metric1 != metric2:
                correlation = abs(CORRELATION_MATRIX[metric1][metric2])
                weight = correlation * normalized[metric1] * normalized[metric2]
                correlation_score += weight
                total_weight += correlation
    
    # Normalize final score
    final_score = correlation_score / total_weight if total_weight > 0 else 0
    
    # Add individual metric bonuses
    bonus_score = (normalized['ROE'] * 0.3 + normalized['ROA'] * 0.2 + 
                   normalized['P/E Ratio'] * 0.2 + normalized['P/B Ratio'] * 0.15 + 
                   normalized['D/E Ratio'] * 0.15)
    
    return (final_score + bonus_score) / 2

# Initialize test data
test_companies = load_test_data()
faq_data = load_faq_data()

# Create and train enhanced model
@st.cache_resource
def create_enhanced_model():
    # Create sample training data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    pe_ratios = np.random.normal(25, 15, n_samples)
    pb_ratios = np.random.normal(5, 3, n_samples)
    de_ratios = np.random.normal(1, 0.8, n_samples)
    roe_values = np.random.normal(20, 10, n_samples)
    roa_values = np.random.normal(8, 5, n_samples)
    
    X = np.column_stack([pe_ratios, pb_ratios, de_ratios, roe_values, roa_values])
    
    # Create correlation-based features
    correlation_scores = []
    for i in range(n_samples):
        metrics = {
            'P/E Ratio': pe_ratios[i],
            'P/B Ratio': pb_ratios[i],
            'D/E Ratio': de_ratios[i],
            'ROE': roe_values[i],
            'ROA': roa_values[i]
        }
        correlation_scores.append(calculate_correlation_score(metrics))
    
    correlation_scores = np.array(correlation_scores)
    
    # Enhanced feature matrix
    X_enhanced = np.column_stack([X, correlation_scores])
    
    # Create target based on correlation score and traditional metrics
    y = ((correlation_scores > 0.4) & (roe_values > 15) & (roa_values > 5) & 
         (pe_ratios < 40) & (de_ratios < 2)).astype(int)
    
    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_enhanced)
    
    model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=10)
    model.fit(X_scaled, y)
    
    return model, scaler

# Load enhanced model
model, scaler = create_enhanced_model()

# Sidebar with FAQ
with st.sidebar:
    st.header("📖 FAQ Section")
    search_query = st.text_input("🔍 Search FAQs")
    
    if search_query:
        filtered_faqs = {k: v for k, v in faq_data.items() if search_query.lower() in k.lower()}
    else:
        filtered_faqs = faq_data
    
    if filtered_faqs:
        selected_question = st.selectbox("Choose a Question", list(filtered_faqs.keys()))
        st.write("**Answer:**", filtered_faqs[selected_question])
    else:
        st.write("No questions match your search.")
    
    # Correlation Matrix Visualization
    st.header("🔗 Correlation Matrix")
    with st.expander("📊 View Correlation Heatmap"):
        # Create correlation matrix DataFrame
        corr_df = pd.DataFrame(CORRELATION_MATRIX)
        
        # Create heatmap
        fig_corr = px.imshow(
            corr_df.values,
            x=corr_df.columns,
            y=corr_df.index,
            color_continuous_scale='RdBu',
            aspect="auto",
            title="Financial Metrics Correlation Matrix"
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Metrics Guide
    st.header("📊 Metrics Guide")
    with st.expander("📘 View Metrics Descriptions"):
        st.write("""
        - **P/E Ratio**: Price-to-Earnings ratio
        - **P/B Ratio**: Price-to-Book ratio  
        - **D/E Ratio**: Debt-to-Equity ratio
        - **ROE**: Return on Equity (%)
        - **ROA**: Return on Assets (%)
        - **Correlation Score**: Composite score based on metric relationships
        """)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Compare Companies",
    "📈 Individual Recommendation", 
    "🔍 Custom Analysis",
    "💬 Chat with AI"
])

with tab1:
    st.header("🔍 Compare Multiple Companies")
    
    company_names = list(test_companies.keys())
    selected_companies = st.multiselect(
        "Choose Companies to Compare", 
        company_names,
        help="Select multiple companies to compare their financial metrics"
    )
    
    if selected_companies:
        if st.button("💡 Get Recommendations for Selected Companies"):
            comparison_results = []
            
            for company in selected_companies:
                metrics = test_companies[company]
                
                # Calculate correlation score
                correlation_score = calculate_correlation_score(metrics)
                
                # Prepare features for prediction
                features = np.array([[
                    metrics['P/E Ratio'],
                    metrics['P/B Ratio'], 
                    metrics['D/E Ratio'],
                    metrics['ROE'],
                    metrics['ROA'],
                    correlation_score
                ]])
                
                # Scale and predict
                scaled_features = scaler.transform(features)
                prediction = model.predict(scaled_features)[0]
                prediction_proba = model.predict_proba(scaled_features)[0]
                
                recommendation = "Buy" if prediction == 1 else "Hold/Sell"
                confidence = prediction_proba[1] * 100 if prediction == 1 else prediction_proba[0] * 100
                
                comparison_results.append({
                    "Company": company,
                    "Recommendation": recommendation,
                    "Confidence (%)": f"{confidence:.1f}%",
                    "Correlation Score": f"{correlation_score:.3f}",
                    "P/E Ratio": metrics['P/E Ratio'],
                    "P/B Ratio": metrics['P/B Ratio'],
                    "D/E Ratio": metrics['D/E Ratio'],
                    "ROE": metrics['ROE'],
                    "ROA": metrics['ROA']
                })
            
            # Display results
            comparison_df = pd.DataFrame(comparison_results)
            st.write("### 📊 Comparison Results")
            st.dataframe(comparison_df, use_container_width=True)
            
            # Create visualization
            fig = px.scatter(
                comparison_df,
                x="Correlation Score",
                y="Confidence (%)",
                color="Recommendation",
                size="ROE",
                hover_data=["Company", "P/E Ratio", "ROA"],
                title="Investment Recommendation Analysis"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Download option
            csv = comparison_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results",
                csv,
                "comparison_results.csv",
                "text/csv"
            )

with tab2:
    st.header("📈 Individual Stock Recommendation")
    
    selected_company = st.selectbox("Choose a Company", list(test_companies.keys()))
    
    if selected_company:
        # Get company metrics
        metrics = test_companies[selected_company]
        correlation_score = calculate_correlation_score(metrics)
        
        # Display current metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("P/E Ratio", f"{metrics['P/E Ratio']}")
        with col2:
            st.metric("P/B Ratio", f"{metrics['P/B Ratio']}")
        with col3:
            st.metric("D/E Ratio", f"{metrics['D/E Ratio']}")
        with col4:
            st.metric("ROE", f"{metrics['ROE']}%")
        with col5:
            st.metric("ROA", f"{metrics['ROA']}%")
        with col6:
            st.metric("Correlation Score", f"{correlation_score:.3f}")
        
        if st.button("💡 Get Recommendation"):
            # Prepare features for prediction
            features = np.array([[
                metrics['P/E Ratio'],
                metrics['P/B Ratio'],
                metrics['D/E Ratio'],
                metrics['ROE'],
                metrics['ROA'],
                correlation_score
            ]])
            
            # Scale and predict
            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)[0]
            prediction_proba = model.predict_proba(scaled_features)[0]
            feature_importance = model.feature_importances_
            
            recommendation = "✅ Buy the stock!" if prediction == 1 else "❌ Hold/Sell the stock."
            confidence = prediction_proba[1] * 100 if prediction == 1 else prediction_proba[0] * 100
            
            # Display recommendation
            color = "green" if prediction == 1 else "red"
            st.markdown(
                f"<h3 style='color:{color}; text-align: center;'>{recommendation}</h3>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<h4 style='text-align: center;'>Confidence: {confidence:.1f}%</h4>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<h4 style='text-align: center;'>Correlation Score: {correlation_score:.3f}</h4>",
                unsafe_allow_html=True
            )
            
            # Feature importance visualization
            feature_names = ['P/E Ratio', 'P/B Ratio', 'D/E Ratio', 'ROE', 'ROA', 'Correlation Score']
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=True)
            
            fig_importance = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance in Prediction"
            )
            st.plotly_chart(fig_importance, use_container_width=True)

with tab3:
    st.header("🔍 Custom Financial Analysis")
    st.write("Enter custom financial metrics to get a recommendation:")
    
    with st.form("custom_analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            custom_pe = st.number_input("P/E Ratio", min_value=0.1, max_value=200.0, value=25.0, step=0.1)
            custom_pb = st.number_input("P/B Ratio", min_value=0.1, max_value=50.0, value=3.0, step=0.1)
            custom_de = st.number_input("D/E Ratio", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        
        with col2:
            custom_roe = st.number_input("ROE (%)", min_value=-50.0, max_value=100.0, value=15.0, step=0.1)
            custom_roa = st.number_input("ROA (%)", min_value=-20.0, max_value=50.0, value=8.0, step=0.1)
        
        submitted = st.form_submit_button("🔍 Analyze Custom Metrics")
    
    if submitted:
        custom_metrics = {
            'P/E Ratio': custom_pe,
            'P/B Ratio': custom_pb,
            'D/E Ratio': custom_de,
            'ROE': custom_roe,
            'ROA': custom_roa
        }
        
        custom_correlation_score = calculate_correlation_score(custom_metrics)
        
        # Prepare features for prediction
        features = np.array([[
            custom_pe, custom_pb, custom_de, custom_roe, custom_roa, custom_correlation_score
        ]])
        
        # Scale and predict
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        prediction_proba = model.predict_proba(scaled_features)[0]
        
        recommendation = "✅ Recommended for Purchase" if prediction == 1 else "❌ Not Recommended"
        confidence = prediction_proba[1] * 100 if prediction == 1 else prediction_proba[0] * 100
        
        # Display results
        color = "green" if prediction == 1 else "red"
        st.markdown(
            f"<h3 style='color:{color}; text-align: center;'>{recommendation}</h3>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h4 style='text-align: center;'>Confidence: {confidence:.1f}%</h4>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h4 style='text-align: center;'>Correlation Score: {custom_correlation_score:.3f}</h4>",
            unsafe_allow_html=True
        )

with tab4:
    st.header("💬 Stock & Finance Assistant")
    
    # Initialize chat history
    if "finance_chat_history" not in st.session_state:
        st.session_state.finance_chat_history = []
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.finance_chat_history:
            role = msg.get("role")
            content = msg.get("content")
            
            if role == "user":
                st.markdown(f"**🧑 You:** {content}")
            else:
                st.markdown(f"**🤖 Assistant:** {content}")
            st.markdown("---")
    
    # Chat input
    user_input = st.text_input("Ask me about stocks, finance, or trading:", key="chat_input")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        send_clicked = st.button("Send")
    with col2:
        if st.button("Clear Chat"):
            st.session_state.finance_chat_history = []
            st.rerun()
    
    # Process user input
    if send_clicked and user_input and user_input.strip():
        # Add user message
        st.session_state.finance_chat_history.append({
            "role": "user",
            "content": user_input.strip()
        })
        
        # Show thinking indicator
        with st.spinner("🤖 Assistant is thinking..."):
            try:
                # API call to Groq
                api_url = "https://api.groq.com/openai/v1/chat/completions"
                api_headers = {
                    "Authorization": "Bearer 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",
                    "Content-Type": "application/json"
                }
                
                # Build conversation
                messages = [
                    {
                        "role": "system",
                        "content": "You are a knowledgeable AI assistant specializing in stocks, finance, trading, and investment advice. You understand correlation analysis and financial metrics relationships. Provide clear, accurate, and actionable information. Keep responses concise but informative."
                    }
                ]
                
                # Add recent chat history
                recent_messages = st.session_state.finance_chat_history[-8:]
                messages.extend(recent_messages)
                
                # API request
                payload = {
                    "model": "llama3-8b-8192",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "top_p": 0.9,
                    "stream": False
                }
                
                response = requests.post(api_url, headers=api_headers, json=payload, timeout=45)
                
                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get("choices") and len(response_data["choices"]) > 0:
                        assistant_reply = response_data["choices"][0]["message"]["content"].strip()
                        if assistant_reply:
                            st.session_state.finance_chat_history.append({
                                "role": "assistant",
                                "content": assistant_reply
                            })
                        else:
                            st.session_state.finance_chat_history.append({
                                "role": "assistant",
                                "content": "I apologize, but I couldn't generate a proper response. Could you please rephrase your question?"
                            })
                    else:
                        st.session_state.finance_chat_history.append({
                            "role": "assistant",
                            "content": "I'm having trouble processing your request right now. Please try again."
                        })
                else:
                    st.session_state.finance_chat_history.append({
                        "role": "assistant",
                        "content": f"⚠️ API Error ({response.status_code}). Please try again later."
                    })
                    
            except requests.exceptions.Timeout:
                st.session_state.finance_chat_history.append({
                    "role": "assistant",
                    "content": "⏰ Request timed out. Please try again."
                })
            except Exception as e:
                st.session_state.finance_chat_history.append({
                    "role": "assistant",
                    "content": f"❌ Error: {str(e)[:100]}"
                })
        
        # Refresh to show new messages
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>NextGen Investors Beta v2.0 - Enhanced with Correlation Analysis - For Educational Purposes Only</div>",
    unsafe_allow_html=True
)
