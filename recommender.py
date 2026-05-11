import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def calcule_similarite_items(df_notes):

    df_items = df_notes.T

    df_items_filled = df_items.replace(0, np.nan).fillna(df_items.mean(axis=1), )

    for film in df_items.index:
        moyenne = df_items.loc[film][df_items.loc[film] != 0].mean()
        if np.isnan(moyenne):
            moyenne = 3.0 
        df_items_filled.loc[film] = df_items.loc[film].replace(0, moyenne)

    matrice_sim = cosine_similarity(df_items_filled)

    df_similarite = pd.DataFrame(
        matrice_sim,
        index=df_notes.columns,
        columns=df_notes.columns
    )

    return df_similarite


def predire_note(utilisateur, film_cible, df_notes, df_similarite, n_voisins=3):
    notes_utilisateur = df_notes.loc[utilisateur]
    films_notes = notes_utilisateur[notes_utilisateur > 0].index.tolist()

    if film_cible in films_notes:
        films_notes.remove(film_cible)

    if len(films_notes) == 0:
        return 0.0 

    similarites = df_similarite.loc[film_cible, films_notes]

    top_voisins = similarites.nlargest(n_voisins)

    numerateur = 0.0
    denominateur = 0.0

    for film_voisin, sim in top_voisins.items():
        note = df_notes.loc[utilisateur, film_voisin]
        numerateur   += sim * note
        denominateur += abs(sim)

    if denominateur == 0:
        return 0.0

    note_predite = numerateur / denominateur

    return round(max(1.0, min(5.0, note_predite)), 2)


def recommander_top_n(utilisateur, df_notes, df_similarite, n=5, n_voisins=3):

    notes_utilisateur = df_notes.loc[utilisateur]
    films_non_vus = notes_utilisateur[notes_utilisateur == 0].index.tolist()

    if len(films_non_vus) == 0:
        return pd.DataFrame(columns=["Film", "Note prédite"])

    
    predictions = []
    for film in films_non_vus:
        note_pred = predire_note(
            utilisateur, film, df_notes, df_similarite, n_voisins
        )
        predictions.append({
            "Film": film,
            "Note prédite": note_pred
        })

    df_reco = pd.DataFrame(predictions)
    df_reco = df_reco.sort_values("Note prédite", ascending=False).head(n)
    df_reco = df_reco.reset_index(drop=True)
    df_reco.index += 1 

    return df_reco


def get_films_similaires(film, df_similarite, n=5):

    similarites = df_similarite[film].drop(index=film)

    top_similaires = similarites.nlargest(n).reset_index()
    top_similaires.columns = ["Film similaire", "Similarité"]
    top_similaires["Similarité"] = top_similaires["Similarité"].round(3)
    top_similaires.index += 1

    return top_similaires
