# streamlit run app2.py


import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Modell laden mit Fehlerabfang
try:
    model = tf.keras.models.load_model("my_model2.keras")
except Exception as e:
    st.error(f"Fehler beim Laden des Modells: {e}")
    st.stop()

# Session State für Vorhersagen initialisieren
if "predictions" not in st.session_state:
    st.session_state.predictions = []

# App-Titel
st.title("🧠 Handschriftliche Ziffern-Erkennung mit MNIST")
st.write("✏️ Zeichne eine Ziffer (0–9) im Feld:")

# Button zum Löschen des Vorhersageverlaufs
if st.button("🗑️ Vorhersagen löschen"):
    st.session_state.predictions = []

# Zeichenfläche (280x280)
canvas_result = st_canvas(
    stroke_color="black",
    stroke_width=15,
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Bild vorbereiten und vorhersagen
if canvas_result.image_data is not None:
    # Umwandlung zu PIL-Bild und Vorverarbeitung
    image = Image.fromarray(canvas_result.image_data.astype(np.uint8))
    image = image.convert("L")  # Graustufen
    image = image.resize((28, 28))  # Größe wie MNIST
    image = np.array(image) / 255.0  # Normalisieren
    image = 1 - image  # Schwarz auf Weiß umkehren (optional, abhängig vom Training)

    # Umformen für Modell: (1, 28, 28, 1)
    image = image.reshape(1, 28, 28, 1).astype(np.float32)

    # Vorhersage mit try/except für stabile Fehlerabfangung
    try:
        predictions = model.predict(image)
        predicted_label = np.argmax(predictions)
        st.success(f"✅ Vorhergesagte Zahl: {predicted_label}")
        st.bar_chart(predictions[0])

        # Vorhersage speichern
        st.session_state.predictions.append(predicted_label)
    except Exception as e:
        st.error(f"❌ Fehler bei der Vorhersage: {e}")

# Vorhersage-Verlauf anzeigen, falls vorhanden
if st.session_state.predictions:
    st.subheader("📊 Vorhersage-Verlauf")
    st.write(st.session_state.predictions)
