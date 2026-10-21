
"""Smart Keyboard V2 — NLP Lab 02."""

import re
import math
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

from src.pipeline import (
    NGramModel,
    EOS,
    load_documents,
    backoff_top_k,
)

st.set_page_config(
    page_title="Smart Keyboard V2",
    page_icon="⌨️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- STYLE --------------------

st.markdown("""
<style>
.block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    background: rgba(120, 150, 190, 0.08);
    border: 1px solid rgba(120, 150, 190, 0.20);
    padding: 16px;
    border-radius: 12px;
}
div[data-testid="stProgressBar"] > div {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("⌨️ Smart Keyboard V2")
st.caption("Next-word prediction · N-gram language models · NLP Lab 02")

# -------------------- SIDEBAR --------------------

st.sidebar.title("⚙️ Configuration")

source = st.sidebar.selectbox(
    "Dataset",
    ["Corpus local du TP", "Hugging Face — Rotten Tomatoes"],
)

if source.startswith("Hugging Face"):
    limit = st.sidebar.slider(
        "Nombre de textes",
        min_value=200,
        max_value=1200,
        value=600,
        step=200,
    )
else:
    limit = 600

order_label = st.sidebar.selectbox(
    "Modèle de prédiction",
    ["Trigram (3-gram)", "Bigram (2-gram)", "Unigram (1-gram)"],
)

order = {
    "Trigram (3-gram)": 3,
    "Bigram (2-gram)": 2,
    "Unigram (1-gram)": 1,
}[order_label]

k = st.sidebar.select_slider(
    "Lissage add-k",
    options=[0.01, 0.1, 0.5, 1.0],
    value=0.1,
)

st.sidebar.caption(
    "Le lissage évite les probabilités nulles "
    "pour les mots et contextes inconnus."
)


# -------------------- DATA & MODELS --------------------

def tokenize(text):
    """Convertir un texte anglais en liste de mots."""
    return re.findall(r"\b\w+\b", text.lower())


@st.cache_resource(show_spinner="Chargement des données et entraînement...")
def prepare_models(source, limit, k):
    if source == "Corpus local du TP":
        docs = load_documents()
        ids = sorted(docs)

        # Séparation par document pour limiter la fuite de données.
        cut = int(len(ids) * 0.8)
        train_ids = ids[:cut]
        test_ids = ids[cut:]

        train = [
            sentence
            for doc_id in train_ids
            for sentence in docs[doc_id]
        ]
        test = [
            sentence
            for doc_id in test_ids
            for sentence in docs[doc_id]
        ]

    else:
        dataset = load_dataset(
            "cornell-movie-review-data/rotten_tomatoes",
            split=f"train[:{limit}]",
        )

        sentences = [
            tokenize(row["text"])
            for row in dataset
            if row["text"] and tokenize(row["text"])
        ]

        cut = int(len(sentences) * 0.8)
        train, test = sentences[:cut], sentences[cut:]

    models = {
        n: NGramModel(train, n=n, k=k)
        for n in (1, 2, 3)
    }

    return models, train, test


def top3_accuracy(model, sentences, max_sentences=100):
    """Précision Top-3 sur un échantillon reproductible."""
    hits = 0
    total = 0

    for sentence in sentences[:max_sentences]:
        # Évaluer le prochain mot à chaque position, EOS compris.
        for i, true_word in enumerate(sentence + [EOS]):
            context = sentence[:i]
            suggestions = model.top_k(context, k=3)
            predicted = [word for word, _ in suggestions]

            truth = model.clean([true_word])[0]
            hits += int(truth in predicted)
            total += 1

    return hits / total if total else 0.0


try:
    models, train, test = prepare_models(source, limit, k)

    # -------------------- OVERVIEW --------------------

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Textes d'entraînement", len(train))
    m2.metric("Textes de test", len(test))
    m3.metric("Taille du vocabulaire", models[3].V)
    m4.metric("Modèle sélectionné", order_label)

    st.divider()

    prediction_tab, evaluation_tab, errors_tab, data_tab = st.tabs([
        "🔮 Prédiction",
        "📊 Évaluation",
        "🔍 Analyse des erreurs",
        "📚 Dataset",
    ])

    # -------------------- PREDICTION --------------------

    with prediction_tab:
        st.subheader("Prédire le prochain mot")
        st.write(
            "Saisis une phrase en anglais. "
            "Le modèle propose les trois mots les plus probables."
        )

        with st.form("prediction_form"):
            text = st.text_input(
                "Ta phrase",
                value="the happy girls",
                placeholder="Type a sentence in English...",
            )
            submitted = st.form_submit_button(
                "✨ Prédire",
                use_container_width=True,
            )

        # Afficher également une prédiction au premier chargement.
        if submitted or text.strip():
            words = tokenize(text)

            if not words:
                st.info("Saisis au moins un mot.")
            else:
                unknown = [
                    word for word in words
                    if word not in models[1].vocab
                ]

                if unknown:
                    st.warning(
                        "Mots inconnus : "
                        + ", ".join(dict.fromkeys(unknown))
                        + ". Ils seront traités comme <unk>."
                    )

                tri, bi, uni = models[3], models[2], models[1]

                suggestions, used_order = backoff_top_k(
                    tri, bi, uni, words, 3
                )

                st.markdown(
                    f"### Top-3 suggestions · Modèle utilisé : {used_order}"
                )

                if not suggestions:
                    st.info("Aucune suggestion disponible.")
                else:
                    cols = st.columns(3)

                    for rank, (word, probability) in enumerate(
                        suggestions, start=1
                    ):
                        label = (
                            "(fin de phrase)"
                            if word == EOS
                            else word
                        )

                        with cols[rank - 1]:
                            st.markdown(f"#### {rank}. {label}")
                            st.metric(
                                "Probabilité",
                                f"{probability:.2%}",
                            )
                            st.progress(
                                min(max(float(probability), 0.0), 1.0)
                            )

                if not tri.seen_context(words):
                    st.info(
                        "Ce contexte n'a pas été observé par le "
                        "trigram. Le back-off peut utiliser un modèle "
                        "d'ordre inférieur."
                    )

        st.caption(
            "Les probabilités sont des estimations statistiques. "
            "Elles ne représentent pas nécessairement les habitudes "
            "réelles de saisie."
        )

    # -------------------- EVALUATION --------------------

    with evaluation_tab:
        st.subheader("Comparer les modèles")

        rows = []

        for n in (1, 2, 3):
            model = models[n]
            perplexity, zero_share = model.perplexity(test)

            rows.append({
                "Modèle": {1: "Unigram", 2: "Bigram", 3: "Trigram"}[n],
                "Ordre": n,
                "Perplexité": perplexity,
                "Top-3 accuracy (échantillon)": top3_accuracy(model, test),
                "Tokens à probabilité nulle": zero_share,
            })

        results = pd.DataFrame(rows)

        display_results = results.copy()
        display_results["Perplexité"] = display_results[
            "Perplexité"
        ].map(
            lambda x: "∞" if math.isinf(x) else f"{x:.3f}"
        )
        display_results["Top-3 accuracy (échantillon)"] = (
            display_results["Top-3 accuracy (échantillon)"]
            .map(lambda x: f"{x:.2%}")
        )
        display_results["Tokens à probabilité nulle"] = (
            display_results["Tokens à probabilité nulle"]
            .map(lambda x: f"{x:.2%}")
        )

        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "La perplexité utilise l'ensemble du jeu de test. "
            "La précision Top-3 est calculée sur les 100 premières "
            "phrases de test au maximum."
        )

        chart_data = results.set_index("Modèle")[
            "Top-3 accuracy (échantillon)"
        ]

        st.markdown("#### Précision Top-3 par modèle")
        st.bar_chart(chart_data)

        st.markdown("#### Perplexité par modèle")
        finite_results = results[
            results["Perplexité"].apply(math.isfinite)
        ].set_index("Modèle")["Perplexité"]

        if not finite_results.empty:
            st.bar_chart(finite_results)

        st.info(
            "En général, une perplexité plus faible est préférable, "
            "tandis qu'une précision Top-3 plus élevée est préférable. "
            "Le meilleur modèle dépend du corpus et du lissage."
        )

    # -------------------- ERROR ANALYSIS --------------------

    with errors_tab:
        st.subheader("Exemples de prédictions")
        st.write(
            "Compare le mot attendu avec les trois suggestions "
            "du modèle sélectionné."
        )

        model = models[order]
        error_rows = []

        for sentence in test[:100]:
            for i, true_word in enumerate(sentence + [EOS]):
                context = sentence[:i]
                guesses = model.top_k(context, k=3)
                predicted = [word for word, _ in guesses]
                truth = model.clean([true_word])[0]

                if truth not in predicted:
                    error_rows.append({
                        "Contexte": " ".join(context[-4:]) or "<début>",
                        "Mot attendu": true_word,
                        "Suggestions": ", ".join(
                            word for word, _ in guesses
                        ),
                    })

                if len(error_rows) >= 30:
                    break

            if len(error_rows) >= 30:
                break

        if error_rows:
            st.dataframe(
                pd.DataFrame(error_rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success(
                "Aucune erreur Top-3 détectée dans l'échantillon."
            )

        st.markdown("**Pourquoi certaines prédictions échouent-elles ?**")
        st.markdown(
            "- **Contexte inconnu :** la séquence n'existait pas dans "
            "les données d'entraînement.\n"
            "- **Données rares :** certaines combinaisons de mots "
            "apparaissent trop peu souvent.\n"
            "- **Limite du modèle :** les N-gram utilisent des "
            "fréquences locales et ne comprennent pas réellement "
            "le sens d'une phrase."
        )

    # -------------------- DATASET --------------------

    with data_tab:
        st.subheader("Informations sur le corpus")

        c1, c2 = st.columns(2)
        c1.metric("Phrases d'entraînement", len(train))
        c2.metric("Phrases de test", len(test))

        st.markdown("#### Exemples du jeu d'entraînement")

        preview = pd.DataFrame({
            "Texte": [" ".join(sentence) for sentence in train[:20]]
        })

        st.dataframe(
            preview,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Exporter l'aperçu CSV",
            data=preview.to_csv(index=False).encode("utf-8"),
            file_name="smart_keyboard_preview.csv",
            mime="text/csv",
        )

        st.caption(
            "Le dataset Rotten Tomatoes contient des critiques de films. "
            "Le corpus synthétique du TP reste disponible pour comparer "
            "avec les résultats de référence du laboratoire."
        )

except Exception as error:
    st.error("Le chargement ou l'entraînement a échoué.")
    st.code(str(error))
    st.markdown(
        "Vérifie les dépendances, le dataset, le réseau et "
        "la structure de `src/pipeline.py`."
    )
