"""Application Streamlit : formulaire praticien + affichage du risque (SP-27, SP-29)."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Risque de diabète", page_icon="🩺", layout="centered")
st.title("🩺 Évaluation du risque de diabète")
st.caption("Outil d'aide à la décision. Il ne remplace pas l'avis d'un médecin.")

with st.form("patient_form"):
    st.subheader("Données du patient")
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Nombre de grossesses", min_value=0, max_value=20, value=0, step=1)
        age = st.number_input("Âge (années)", min_value=1, max_value=120, value=30, step=1)
        dpf = st.number_input(
            "Fonction de pedigree du diabète (DPF)",
            min_value=0.01, max_value=3.0, value=0.5, step=0.01,
            help="Score d'antécédents familiaux de diabète.",
        )
        bmi = st.number_input("IMC (BMI)", min_value=0.0, max_value=70.0, value=0.0, step=0.1,
                              help="0 = inconnu")

    with col2:
        glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=0.0, step=1.0,
                                  help="0 = inconnu")
        blood_pressure = st.number_input("Pression artérielle (mm Hg)", min_value=0.0, max_value=200.0,
                                         value=0.0, step=1.0, help="0 = inconnu")
        skin_thickness = st.number_input("Épaisseur du pli cutané (mm)", min_value=0.0, max_value=100.0,
                                         value=0.0, step=1.0, help="0 = inconnu")
        insulin = st.number_input("Insuline (mu U/ml)", min_value=0.0, max_value=900.0, value=0.0, step=1.0,
                                  help="0 = inconnu")

    st.info("Les champs à 0 sont traités comme « inconnus » et estimés automatiquement.")
    submitted = st.form_submit_button("Évaluer le risque", use_container_width=True)

if submitted:
    payload = {
        "Pregnancies": int(pregnancies),
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": int(age),
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    except requests.exceptions.ConnectionError:
        st.error("Impossible de joindre l'API. Vérifiez qu'elle est démarrée.")
        st.stop()

    if response.status_code == 422:
        detail = response.json().get("detail")
        st.warning(detail if isinstance(detail, str) else "Valeurs invalides. Vérifiez les champs saisis.")
    elif response.status_code != 200:
        st.error(f"Erreur de l'API (code {response.status_code}).")
    else:
        result = response.json()
        proba = result["probability_high_risk"]

        st.subheader("Résultat")
        if result["risk_category"] == "risque élevé":
            st.error("🔴 RISQUE ÉLEVÉ")
            st.markdown(
                "**Conseils :**\n"
                "- Consulter un médecin pour un bilan glycémique (glycémie à jeun, HbA1c).\n"
                "- Adapter l'alimentation (limiter sucres rapides et produits ultra-transformés).\n"
                "- Pratiquer une activité physique régulière.\n"
                "- Prévoir un suivi médical régulier."
            )
        else:
            st.success("🟢 RISQUE FAIBLE")
            st.markdown(
                "**Conseils :**\n"
                "- Maintenir une alimentation équilibrée et une activité physique régulière.\n"
                "- Contrôler sa glycémie lors des bilans de santé réguliers."
            )

        st.progress(min(max(proba, 0.0), 1.0), text=f"Probabilité de risque élevé : {proba:.0%}")
