import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile

# =========================
# Chargement des modèles
# =========================

fire_model = YOLO("best_fire.pt")
accident_model = YOLO("best_accident.pt")
traffic_model = YOLO("yolov8n.pt")

# =========================
# Interface
# =========================

st.set_page_config(
    page_title="Smart City AI Dashboard",
    layout="wide"
)

st.title("🚦 Smart City AI Dashboard")

st.write(
    "Détection des accidents, incendies et analyse du trafic en temps réel"
)

uploaded_file = st.file_uploader(
    "Choisir une vidéo",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_file.name)

    # Colonnes statistiques
    col1, col2, col3, col4 = st.columns(4)

    vehicle_metric = col1.empty()
    traffic_metric = col2.empty()
    accident_metric = col3.empty()
    fire_metric = col4.empty()

    stframe = st.empty()

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        annotated_frame = frame.copy()

        # =========================
        # Prédictions
        # =========================

        fire_results = fire_model(frame, verbose=False)

        accident_results = accident_model(
            frame,
            verbose=False
        )

        traffic_results = traffic_model(
            frame,
            verbose=False
        )

        # =========================
        # Compteurs
        # =========================

        vehicle_count = 0
        accident_count = 0
        fire_count = 0

        # =========================
        # FIRE / SMOKE
        # =========================

        for box in fire_results[0].boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            cls = int(box.cls[0])

            label = fire_model.names[cls]

            fire_count += 1

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        # =========================
        # ACCIDENT
        # =========================

        for box in accident_results[0].boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            accident_count += 1

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                "ACCIDENT",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        # =========================
        # TRAFIC
        # =========================

        for box in traffic_results[0].boxes:

            cls = int(box.cls[0])

            label = traffic_model.names[cls]

            if label in [
                "car",
                "truck",
                "bus",
                "motorcycle"
            ]:

                vehicle_count += 1

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # =========================
        # Analyse trafic
        # =========================

        if vehicle_count < 10:
            traffic_status = "Faible"

        elif vehicle_count < 25:
            traffic_status = "Moyen"

        else:
            traffic_status = "Dense"

        # =========================
        # Dashboard
        # =========================

        vehicle_metric.metric(
            "🚗 Véhicules",
            vehicle_count
        )

        traffic_metric.metric(
            "🚦 Trafic",
            traffic_status
        )

        accident_metric.metric(
            "🚨 Accidents",
            accident_count
        )

        fire_metric.metric(
            "🔥 Fire/Smoke",
            fire_count
        )

        # =========================
        # Affichage vidéo
        # =========================

        annotated_frame = cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB
        )

        stframe.image(
            annotated_frame,
            channels="RGB",
            use_container_width=True
        )

    cap.release()

    st.success(
        "✅ Analyse terminée"
    )