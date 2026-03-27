import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, movies_path: str, tags_path: str):
        self.movies_path = movies_path
        self.tags_path = tags_path
        self.movies_df = None
        self.tfidf_matrix = None
        self.similarity_matrix = None

    def load_and_prepare_data(self) -> None:
        # Load datasets
        movies = pd.read_csv(self.movies_path)
        tags = pd.read_csv(self.tags_path)

        # Keep only needed columns
        tags = tags[["movieId", "tag"]].dropna()

        # Clean tags
        tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()

        # Group tags by movieId into one string
        grouped_tags = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()
        grouped_tags.rename(columns={"tag": "all_tags"}, inplace=True)

        # Clean genres
        movies["genres"] = movies["genres"].fillna("").astype(str)
        movies["clean_genres"] = (
            movies["genres"]
            .str.replace("|", " ", regex=False)
            .str.lower()
            .str.strip()
        )

        # Merge movies with grouped tags
        movies = movies.merge(grouped_tags, on="movieId", how="left")
        movies["all_tags"] = movies["all_tags"].fillna("")

        # Create final text column
        movies["combined_features"] = movies["clean_genres"] + " " + movies["all_tags"]

        # Remove extra spaces
        movies["combined_features"] = movies["combined_features"].str.replace(r"\s+", " ", regex=True).str.strip()

        self.movies_df = movies
        print("Movies loaded:", len(self.movies_df))


    def build_similarity_model(self) -> None:
        vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = vectorizer.fit_transform(self.movies_df["combined_features"])

    def get_recommendations(self, movie_title: str, top_n: int = 5) -> pd.DataFrame:
        if self.movies_df is None or self.tfidf_matrix is None:
            raise ValueError("Data or model not prepared. Run load_and_prepare_data() and build_similarity_model() first.")

        # Case-insensitive title matching
        matches = self.movies_df[
        self.movies_df["title"].str.lower().str.contains(movie_title.lower(), na=False)]

        if matches.empty:
            raise ValueError(f"Movie '{movie_title}' not found in dataset.")

        movie_index = matches.index[0]

        similarity_scores = cosine_similarity(self.tfidf_matrix[movie_index], self.tfidf_matrix).flatten()
        similarity_scores = list(enumerate(similarity_scores))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Skip the first one because it's the same movie
        top_matches = similarity_scores[1:top_n + 1]
        recommended_indices = [idx for idx, _ in top_matches]

        return self.movies_df.loc[recommended_indices, ["movieId", "title", "genres"]].reset_index(drop=True)