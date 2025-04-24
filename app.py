import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import streamlit as st

# Load datasets
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = pd.read_csv('tmdb_5000_movies.csv')

# Clean credits data
credits_col_renamed = credits.rename(columns={'movie_id': 'id'})

# Convert 'id' columns to numeric for merging
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')

# Merge the two datasets
mv_merge = movies.merge(credits_col_renamed, on='id')

# Clean the merged DataFrame by dropping unnecessary columns
movies_cleaned_df = mv_merge.drop(columns=['homepage','status','title_x','title_y','production_countries'])

# Fill missing overviews with empty string
movies_cleaned_df['overview'] = movies_cleaned_df['overview'].fillna('')

# Initialize TF-IDF Vectorizer
tfv = TfidfVectorizer(min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words='english')

# Apply TF-IDF on the movie overviews
tfv_matrix = tfv.fit_transform(movies_cleaned_df['overview'])

# Compute sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

# Generate movie indices for lookups (convert titles to lowercase)
indices = pd.Series(movies_cleaned_df.index, index=movies_cleaned_df['original_title'].str.lower()).to_dict()

# Define function for recommendations
def recommend_movie(movie_title, top_n=10):
    # Convert input movie title to lowercase for case-insensitive matching
    movie_title = movie_title.lower()
    
    if movie_title not in indices:
        return "Sorry, the movie was not found in the database."
    
    idx = indices[movie_title]
    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1:top_n+1]
    movie_indices = [i[0] for i in sig_scores]
    
    recommendations = movies_cleaned_df[['original_title', 'overview', 'vote_average']].iloc[movie_indices]
    return recommendations

# Streamlit UI
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# Title and introduction
st.title("🎬 Movie Recommendation System 🍿")
st.markdown("""
    ## Get Personalized Movie Recommendations
    Enter a movie title below, and I'll suggest similar movies based on your input. 
    Enjoy discovering new movies!
""")

# Movie input field with styling
movie_input = st.text_input("Enter a Movie Title", placeholder="E.g., Avatar", max_chars=100)

if movie_input:
    st.markdown(f"### Recommendations based on: **{movie_input}**")

    # Get recommendations
    recommendations = recommend_movie(movie_input)

    if isinstance(recommendations, str):  # If no recommendations found
        st.error(recommendations)
    else:
        # Display recommendations in a clean format
        for idx, row in recommendations.iterrows():
            with st.container():
                st.subheader(f"{row['original_title']} ({row['vote_average']}/10)")
                st.write(row['overview'])
                st.markdown("---")  # Horizontal line
st.markdown("--crafted by Srija❤️") 

# Style and layout improvements
st.markdown("""
    <style>
        .css-ffhzg2 {
            text-align: center;
        }
        .stTextInput > div > input {
            font-size: 18px;
            padding: 12px;
            width: 100%;
            border-radius: 8px;
        }
        .stButton>button {
            font-size: 16px;
            background-color: #007BFF;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

