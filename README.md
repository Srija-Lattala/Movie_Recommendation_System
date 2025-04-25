# Movie Recommendation System

[![Streamlit Deployment](https://img.shields.io/badge/Streamlit-Live_App-brightgreen)](https://movie-recommendation-a.streamlit.app/)

This is a Movie Recommendation System built using a dataset of movies and their associated information, including credits and overviews. The system utilizes content-based filtering using a TF-IDF vectorizer and a sigmoid kernel to recommend similar movies based on a given movie title.

## Features
- Recommend similar movies based on a movie title.
- Displays the movie title, overview, and vote average for recommended movies.
- Built using Python, Pandas, Scikit-learn, and Streamlit for the UI.

## Technologies Used
- **Python**: Programming language used for building the system.
- **Pandas**: For data manipulation and cleaning.
- **Scikit-learn**: For creating the TF-IDF vectorizer and calculating similarity.
- **Streamlit**: For building the interactive web UI.

## Requirements

To run this project, you need to have Python and the necessary dependencies installed. You can install the required packages using `pip`:

1. Clone the repository:

   ```bash
   git clone https://github.com/Srija-Lattala/movie_recommendation.git
   cd movie_recommendation
   ```
2. Install Dependencies
  ```bash
  pip install -r requirements.txt
  ```
3.Run the app:
  ```bash
  streamlit run app.py
  ```

## Dataset
The system uses two datasets:

- **tmdb_5000_credits.csv**: Contains the movie credits including movie_id and cast/crew information.
- **tmdb_5000_movies.csv**: Contains movie details such as title, overview, genres, and vote averages.

## How it Works
The recommendation system works by using a content-based filtering approach:

1. The movie overviews are vectorized using a **TF-IDF** (Term Frequency-Inverse Document Frequency) vectorizer.
2. The similarity between movies is calculated using a **sigmoid kernel**.
3. The system recommends movies by finding those with the highest similarity to the given input movie.

## License
This project is open-source and available under the **MIT License**.

## Acknowledgements
- Movie data is sourced from the **TMDb dataset**.
- The recommendation system is based on **content-based filtering** techniques commonly used in information retrieval and recommendation systems.

