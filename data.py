import pandas as pd

def get_ratings_matrix():

    data = {
        "Titanic":        [5, 4, 0, 2, 0, 4, 0, 3],
        "Avatar":         [4, 0, 4, 3, 2, 0, 5, 0],
        "Inception":      [0, 5, 5, 0, 4, 3, 4, 5],
        "Interstellar":   [0, 4, 5, 0, 5, 0, 3, 4],
        "The Matrix":     [2, 0, 4, 0, 5, 2, 5, 4],
        "Forrest Gump":   [5, 5, 0, 5, 0, 5, 0, 3],
        "The Dark Knight":[0, 4, 3, 0, 4, 0, 5, 5],
        "La La Land":     [4, 3, 0, 5, 0, 4, 0, 2],
        "Parasite":       [0, 3, 4, 0, 3, 2, 4, 5],
        "Toy Story":      [3, 4, 0, 5, 0, 5, 0, 3],
    }

    utilisateurs = [
        "Alice", "Bob", "Carol", "David",
        "Eva",   "Frank", "Grace", "Hugo"
    ]

    df = pd.DataFrame(data, index=utilisateurs)

    return df
