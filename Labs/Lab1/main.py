from recommender import MovieRecommender

def main():
    recommender = MovieRecommender(
        movies_path="../../datasets/ml-latest/movies.csv",
        tags_path="../../datasets/ml-latest/tags.csv"
    )
    print("Loading and preparing data...")
    recommender.load_and_prepare_data()

    recommender.build_similarity_model()

    movie_title = input("Enter a movie title: ").strip()

    try:
        recommendations = recommender.get_recommendations(movie_title, top_n=5)
        print("\nRecommended movies:")
        print(recommendations.to_string(index=False))
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()