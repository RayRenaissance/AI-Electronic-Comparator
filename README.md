# AI Product Comparator for Electronics

## Project Overview
An intelligent web application that compares electronics products (like smartphones, laptops, etc.) based on **features**, **price**, **ratings**, and **specifications** using Natural Language Processing (TF-IDF + Cosine Similarity).. Built with Streamlit for the frontend and scikit-learn for AI features.

## Features
- **Product Comparison**: Side-by-side comparison of two electronics products
- **AI-Powered Analysis**: Uses TF-IDF and cosine similarity for feature comparison
- **Smart Recommendations**: AI suggests better products based on multiple factors
- **Filtering System**: Filter products by category and price range
- **Interactive UI**: User-friendly Streamlit interface

## Technologies Used
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **AI/ML**: scikit-learn (TF-IDF, Cosine Similarity)
- **Data Storage**: CSV file

## Installation
1. Clone the repository: `Soon Added`
2. Install dependencies: `pip install -r requirements.txt` if encountered error while installing dependencies please update your pip `python -m pip install --upgrade pip`
3. Run the application: `streamlit run app.py`
4. Make sure your dataset is placed at: data/products.csv

## Setup Instructions

1. **Create virtual environment**
   ```bash
   python -m venv comparator_env
   
   # Activate it:
   # Windows:
   comparator_env\Scripts\activate
   # Mac/Linux:
   source comparator_env/bin/activate
   # Deactivate it:
   deactivate

## ⚙️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/RayRenaissance/AI_Product_Comparator.git
   cd AI_Product_Comparator


## Approach
1. **Data Management**: CSV-based product database with structured specifications
2. **NLP Processing**: TF-IDF vectorization of product specifications
3. **Similarity Analysis**: Cosine similarity for feature comparison
4. **Recommendation Engine**: Rule-based AI with weighted scoring system

## Challenges Faced
1. **Specification Parsing**: Handling unstructured specification text
2. **Similarity Measurement**: Creating meaningful comparison metrics
3. **Recommendation Logic**: Balancing multiple factors (price, rating, features)
4. **UI/UX Design**: Presenting complex comparisons in simple format

## Sample Output
The application provides:
- Side-by-side product comparison
- Feature similarity scores
- AI recommendations with confidence scores
- Key insights and differences

## 🚀 Future Improvements

- 🔍 **LLM Integration:**  
  Add a Large Language Model (like LLaMA, Mistral, or OpenAI GPT) to allow natural language queries such as  
  *“Compare the latest Samsung and OnePlus smartphones under ₹50,000.”*  
  The LLM will interpret user intent and select matching products automatically.

- 📚 **RAG (Retrieval-Augmented Generation):**  
  Use a vector database (like FAISS or Chroma) to store and retrieve product specs dynamically.  
  The LLM can then generate comparison summaries backed by retrieved context for **factually accurate answers**.

- 🛍️ **Smartphone Search & Dynamic Comparison:**  
  Let users type queries like:  
  *“Show me phones with 12GB RAM and 5000mAh battery.”*  
  Your system fetches relevant products from the dataset or web APIs and compares them automatically.

- 🖼️ **Image-Based Comparison (Vision + LLM):**  
  Allow uploading product images and extract specs via OCR + vision models, then perform comparisons.

- 🌐 **Real-time Data Integration:**  
  Connect APIs like Amazon Product API, Flipkart API, or GSM Arena scraping (free APIs available) to fetch live data — price, stock, and ratings — for up-to-date comparisons.

- 🧠 **Explainable AI Insights:**  
  Show which specs/features contributed most to the AI’s recommendation (feature attribution visualization).

