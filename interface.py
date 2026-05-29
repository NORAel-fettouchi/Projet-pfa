import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
# streamlit run interface.py
# Charger modèle
model = YOLO("best_accident.pt")

# Titre interface
st.title("🚦 Smart City AI Dashboard")

st.write("Détection d'accidents et analyse trafic")

# Upload vidéo
uploaded_file = st.file_uploader(
    "Choisir une vidéo",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:

    # Sauvegarder vidéo temporairement
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    # Ouvrir vidéo
    cap = cv2.VideoCapture(temp_file.name)

    # Affichage vidéo
    stframe = st.empty()

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # Détection YOLO
        results = model(frame)

        # Dessiner résultats
        annotated_frame = results[0].plot()

        # Convertir BGR → RGB
        annotated_frame = cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB
        )

        # Afficher frame
        stframe.image(
            annotated_frame,
            channels="RGB",
            use_container_width=True
        )

    cap.release()

    st.success("✅ Analyse terminée")