import numpy as np

# Matrix of ratings
# Rows = viewers (1 → 5)
# Columns = movies (1 → 3)
movie_ratings = np.array([
    [5, 1, 4],  # Viewer 1
    [4, 4, 2],  # Viewer 2
    [4, 3, 5],  # Viewer 3
    [1, 1, 5],  # Viewer 4
    [3, 2, 1]  # Viewer 5
])

# 1. Average Rating Calculation (per movie → column-wise)
average_ratings = np.mean(movie_ratings, axis=0)
print(f"Average Ratings: {average_ratings}\n")

# 2. Viewer Preference Analysis
# argmax gives index of max element per row (per viewer)
viewer_preferences = np.argmax(movie_ratings, axis=1) + 1  # +1 to match movie numbers
print(f"Viewer Preferences: {viewer_preferences}\n")

# Display results
print("=== Movie Ratings Analysis ===")
for i, avg in enumerate(average_ratings, start=1):
    print(f"Movie {i} - Average Rating: {avg:.2f}")

print("\nViewer Preferences:")
for viewer, movie in enumerate(viewer_preferences, start=1):
    print(f"Viewer {viewer} → Highest rated: Movie {movie}")
