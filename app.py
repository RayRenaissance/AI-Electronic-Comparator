import streamlit as st
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configuration
DATA_PATH = 'data/products.csv'

# Data Loading and Preprocessing
def load_data():
    """
    Load product data from CSV file with proper error handling
    Returns: pandas DataFrame containing product information
    """
    # Check if file exists
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ Dataset not found at: {DATA_PATH}")
        st.info("📋 Please ensure 'products.csv' is placed in the 'data/' folder")
        st.info("💡 You can use the sample dataset provided in the repository")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Validate required columns
        required_columns = ['name', 'category', 'price', 'rating', 'reviews', 'specifications']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            st.info(f"✅ Required columns: {', '.join(required_columns)}")
            return pd.DataFrame()
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return pd.DataFrame()
    
def preprocess_specifications(specs_text):
    """Clean and preprocess specification text"""
    if pd.isna(specs_text):
        return ""
    
    # Convert to lowercase and remove extra spaces
    cleaned = re.sub(r'\s+', ' ', str(specs_text).lower().strip())
    return cleaned

# AI Component
def calculate_feature_similarity(product1_specs, product2_specs):
    """Calculate similarity between two products using TF-IDF and cosine similarity"""
    if not product1_specs or not product2_specs:
        return 0.0
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    try:
        # Fit and transform the specifications
        tfidf_matrix = vectorizer.fit_transform([product1_specs, product2_specs])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix.getrow(0), tfidf_matrix.getrow(1))
        return float(similarity[0][0])
        
    except ValueError as e:
        st.warning(f"⚠️ Similarity calculation failed: {e}")
        return 0.0
    
# Extract numeric values from specifications for quantitative comparison
def extract_numeric_values(specs_text):
    """Extract numeric features from specification text"""
    numeric_features = {}

    # Common patterns for electronics specifications
    patterns = {
        'storage_gb': r'(\d+)\s*gb',
        'ram_gb': r'(\d+)\s*gb ram',
        'battery_mah': r'(\d+)\s*mah',
        'display_inch': r'(\d+\.?\d*)\s*inch',
        'camera_mp': r'(\d+)\s*mp',
        'price': r'\$(\d+)'
    }
    
    for feature, pattern in patterns.items():
        match = re.search(pattern, specs_text.lower())
        if match:
            numeric_features[feature] = float(match.group(1))
    
    return numeric_features

# AI-based recommendation system to suggest better product
def recommend_better_product(product1, product2, df):
    """Generate AI-powered product recommendation"""
    # Check if products exist
    p1_match = df[df['name'] == product1]
    p2_match = df[df['name'] == product2]
    
    if p1_match.empty or p2_match.empty:
        return "Error: Product not found in database", 0.0
    
    p1_data = p1_match.iloc[0]
    p2_data = p2_match.iloc[0]
    
    score_p1 = 0
    score_p2 = 0
    
    # Price comparison (lower is better)
    if p1_data['price'] < p2_data['price']:
        score_p1 += 3
    else:
        score_p2 += 3
    
    # Rating comparison (higher is better)
    if p1_data['rating'] > p2_data['rating']:
        score_p1 += 3
    else:
        score_p2 += 3
    
    # Reviews count comparison (higher is better)
    if p1_data['reviews'] > p2_data['reviews']:
        score_p1 += 2
    else:
        score_p2 += 2

    # Feature similarity analysis
    similarity_score = calculate_feature_similarity(
        preprocess_specifications(p1_data['specifications']),
        preprocess_specifications(p2_data['specifications'])
    )
    
    if similarity_score > 0.7:
        # Products are similar, prioritize price and rating
        if score_p1 > score_p2:
            return f"🎯 AI Recommendation: {product1} is better value for money!", score_p1/(score_p1+score_p2)
        else:
            return f"🎯 AI Recommendation: {product2} is better value for money!", score_p2/(score_p1+score_p2)
    else:
        # Products are different, provide contextual recommendation
        if "camera" in str(p1_data['specifications']).lower() and "camera" in str(p2_data['specifications']).lower():
            p1_cam = extract_numeric_values(str(p1_data['specifications'])).get('camera_mp', 0)
            p2_cam = extract_numeric_values(str(p2_data['specifications'])).get('camera_mp', 0)
            if p1_cam > p2_cam:
                return f"📷 AI Recommendation: {product1} has better camera ({p1_cam}MP vs {p2_cam}MP)!", 0.6
            elif p2_cam > p1_cam:
                return f"📷 AI Recommendation: {product2} has better camera ({p2_cam}MP vs {p1_cam}MP)!", 0.6
    
    return "⚖️ AI Recommendation: Both products have their strengths. Choose based on your specific needs.", 0.5


# Streamlit UI Components
def display_product_card(product_data, color_scheme):
    """Display product information in a styled card"""
    with st.container():
        st.markdown(f"""
        <div style="border: 2px solid {color_scheme}; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #1e1e1e;">
            <h3 style="color: {color_scheme}; margin-top: 0;">{product_data['name']}</h3>
            <p style="color: #ffffff;"><strong>Category:</strong> {product_data['category']}</p>
            <p style="color: #ffffff;"><strong>Price:</strong> ${product_data['price']}</p>
            <p style="color: #ffffff;"><strong>Rating:</strong> ⭐ {product_data['rating']}/5 ({product_data['reviews']} reviews)</p>
            <p style="color: #ffffff;"><strong>Specifications:</strong><br>{str(product_data['specifications']).replace(',', '<br>')}</p>
        </div>
        """, unsafe_allow_html=True)

def display_comparison_table(product1_data, product2_data):
    """Create a comparison table for two products"""
    comparison_data = {
        'Feature': ['Name', 'Category', 'Price', 'Rating', 'Reviews', 'Key Specifications'],
        'Product 1': [
            product1_data['name'],
            product1_data['category'],
            f"${product1_data['price']}",
            f"{product1_data['rating']}/5 ⭐",
            product1_data['reviews'],
            product1_data['specifications']
        ],
        'Product 2': [
            product2_data['name'],
            product2_data['category'],
            f"${product2_data['price']}",
            f"{product2_data['rating']}/5 ⭐",
            product2_data['reviews'],
            product2_data['specifications']
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)

# Main Application
def main():
    """
    Main function to run the Streamlit application
    Sets up the UI and handles user interactions
    """
    # Configure page settings
    st.set_page_config(
        page_title="AI Product Comparator",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Application header
    st.title("🤖 AI Product Comparator for Electronics")
    st.markdown("Compare electronics products based on features, price, and ratings using AI-powered analysis")
    
    # Load product data
    df = load_data()
    if df.empty:
        st.warning("⚠️ No product data available. Please check your dataset.")
        return
    
    # Sidebar for filters and controls
    st.sidebar.header("🔍 Filter Products")
    
    # Category filter
    categories = ['All'] + sorted(list(df['category'].unique()))
    selected_category = st.sidebar.selectbox("Select Category", categories)
    
    # Price range filter
    price_range = st.sidebar.slider(
        "Price Range ($)",
        min_value=int(df['price'].min()),
        max_value=int(df['price'].max()),
        value=(int(df['price'].min()), int(df['price'].max()))
    )
    
    # Filter data based on selections
    filtered_df = df.copy()
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
    
    filtered_df = filtered_df[
        (filtered_df['price'] >= price_range[0]) & 
        (filtered_df['price'] <= price_range[1])
    ]
    
    # Product selection
    st.sidebar.header("🔄 Compare Products")
    product_list = filtered_df['name'].tolist()
    
    if len(product_list) < 2:
        st.warning("⚠️ Not enough products to compare. Adjust your filters.")
        st.info(f"📊 Total products available: {len(df)}")
        return
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        product1 = st.selectbox("Select Product 1", product_list, key='product1')
    with col2:
        # Remove selected product1 from product2 options
        product2_options = [p for p in product_list if p != product1]
        product2 = st.selectbox("Select Product 2", product2_options, key='product2')
    
    # Main comparison area
    if product1 and product2:
        st.header("📊 Product Comparison")
        
        # Get product data
        product1_data = df[df['name'] == product1].iloc[0]
        product2_data = df[df['name'] == product2].iloc[0]
        
        # Display products side by side
        col1, col2 = st.columns(2)
        
        with col1:
            display_product_card(product1_data, "#1f77b4")
        
        with col2:
            display_product_card(product2_data, "#ff7f0e")
        
        # Display comparison table
        st.subheader("📋 Detailed Comparison")
        display_comparison_table(product1_data, product2_data)
        
        # AI-powered analysis and recommendation
        st.subheader("🤖 AI Analysis")
        
        # Calculate similarity
        similarity = calculate_feature_similarity(
            preprocess_specifications(product1_data['specifications']),
            preprocess_specifications(product2_data['specifications'])
        )
        
        # Display similarity score
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Feature Similarity Score", f"{similarity:.2%}")
        
        # Get AI recommendation
        recommendation, confidence = recommend_better_product(product1, product2, df)
        
        with col2:
            st.metric("AI Confidence Score", f"{confidence:.2%}")
        
        # Display recommendation
        st.info(recommendation)
        
        # Additional insights
        st.subheader("💡 Key Insights")
        
        insights = []
        
        # Price comparison insight
        price_diff = abs(product1_data['price'] - product2_data['price'])
        if price_diff > 0:
            cheaper_product = product1 if product1_data['price'] < product2_data['price'] else product2
            insights.append(f"💵 **Price Difference**: {cheaper_product} is ${price_diff} cheaper")
        
        # Rating comparison insight
        rating_diff = abs(product1_data['rating'] - product2_data['rating'])
        if rating_diff > 0:
            higher_rated = product1 if product1_data['rating'] > product2_data['rating'] else product2
            insights.append(f"⭐ **Rating Difference**: {higher_rated} has {rating_diff:.1f} higher rating")
        
        # Reviews comparison insight
        reviews_diff = abs(product1_data['reviews'] - product2_data['reviews'])
        if reviews_diff > 50:
            more_reviews = product1 if product1_data['reviews'] > product2_data['reviews'] else product2
            insights.append(f"📝 **Popularity**: {more_reviews} has {reviews_diff} more reviews")
        
        # Display insights
        for insight in insights:
            st.write(insight)
        
        # Footer
        st.markdown("---")
        st.markdown("*AI recommendations are based on TF-IDF similarity analysis, price, ratings, and reviews.*")


# Run the application
if __name__ == "__main__":
    main()

