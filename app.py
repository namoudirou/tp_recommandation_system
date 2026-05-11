"""
app.py - Interface Streamlit du système de recommandation

Pour lancer l'application :
    streamlit run app.py

Dépendances à installer :
    pip install streamlit pandas numpy scikit-learn
"""

import streamlit as st
import pandas as pd
import numpy as np

from data import get_ratings_matrix
from recommender import (
    calcule_similarite_items,
    recommander_top_n,
    get_films_similaires
)

st.set_page_config(
    page_title="Système de Recommandation",
    page_icon="🎬",
    layout="wide"
)

st.title("Système de Recommandation")
st.subheader("Filtrage Collaboratif Item-Item")

st.markdown("""
> **Principe :** On compare les films entre eux selon les notes des utilisateurs.
> Pour chaque utilisateur, on recommande des films similaires à ceux qu'il a aimés.
""")

@st.cache_data
def charger_donnees():

    df_notes      = get_ratings_matrix()
    df_similarite = calcule_similarite_items(df_notes)
    return df_notes, df_similarite

df_notes, df_similarite = charger_donnees()


with st.expander("Voir la matrice de notes (utilisateurs × films)", expanded=False):
    st.markdown("Les **0** signifient que l'utilisateur n'a pas noté ce film.")

    def colorier_notes(val):
        if val == 0:
            return "background-color: #f0f0f0; color: #aaa"
        elif val >= 4:
            return "background-color: #c8f7c5"  
        elif val >= 3:
            return "background-color: #fff3cd" 
        else:
            return "background-color: #f8d7da" 

    st.dataframe(df_notes.style.map(colorier_notes))

with st.expander("Voir la matrice de similarité cosinus (films × films)", expanded=False):
    st.markdown("""
    Chaque cellule indique à quel point deux films sont similaires (de 0 à 1).
    Plus la valeur est proche de 1, plus les utilisateurs les ont notés de manière similaire.
    """)
    st.dataframe(
        df_similarite.round(3).style.background_gradient(cmap="Blues", vmin=0, vmax=1)
    )

st.divider()
st.header("Recommandations personnalisées")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    utilisateur_choisi = st.selectbox(
        "Choisir un utilisateur",
        options=df_notes.index.tolist(),
        help="Sélectionne l'utilisateur pour qui générer des recommandations"
    )

with col2:
    n_recommandations = st.slider(
        "Nombre de recommandations (N)",
        min_value=1, max_value=5, value=3,
        help="Combien de films recommander ?"
    )

with col3:
    n_voisins = st.slider(
        "Nombre de voisins",
        min_value=1, max_value=5, value=3,
        help="Combien de films similaires utiliser pour prédire la note ?"
    )

if st.button("Générer les recommandations", type="secondary"):

    notes_user = df_notes.loc[utilisateur_choisi]
    films_vus  = notes_user[notes_user > 0]

    col_gauche, col_droite = st.columns(2)

    with col_gauche:
        st.subheader(f"Films notés par {utilisateur_choisi}")
        df_vus = pd.DataFrame({
            "Film": films_vus.index,
            "Note": films_vus.values
        }).reset_index(drop=True)
        df_vus.index += 1

        df_vus["Note"] = df_vus["Note"].apply(lambda n: "⭐" * int(n))
        st.dataframe(df_vus[["Film", "Note", "Note ⭐"]])

    with col_droite:
        st.subheader(f"Top-{n_recommandations} recommandations")

        df_reco = recommander_top_n(
            utilisateur_choisi,
            df_notes,
            df_similarite,
            n=n_recommandations,
            n_voisins=n_voisins
        )

        if df_reco.empty:
            st.info("Cet utilisateur a déjà noté tous les films !")
        else:

            df_reco["Score (sur 5)"] = df_reco["Note prédite"].apply(
                lambda x: f"{x:.2f} / 5.0"
            )
            st.dataframe(df_reco[["Film", "Score (sur 5)"]])

            st.bar_chart(
                df_reco.set_index("Film")["Note prédite"],
                use_container_width=True
            )


st.divider()
st.header("🔍 Films similaires")

film_choisi = st.selectbox(
    "Choisir un film",
    options=df_notes.columns.tolist(),
    help="Voir quels films sont les plus similaires à ce film"
)

n_sim = st.slider("Nombre de films similaires", min_value=1, max_value=9, value=4)

df_sim = get_films_similaires(film_choisi, df_similarite, n=n_sim)

col_a, col_b = st.columns([1, 1])

with col_a:
    st.markdown(f"**Films les plus similaires à _{film_choisi}_**")
    st.dataframe(df_sim)

with col_b:
    st.markdown("**Similarité cosinus**")
    st.bar_chart(df_sim.set_index("Film similaire")["Similarité"])

