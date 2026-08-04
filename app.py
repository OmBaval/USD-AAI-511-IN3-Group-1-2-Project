import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import tempfile
import os

from tensorflow.keras.preprocessing.sequence import pad_sequences
from music21 import converter, instrument, note, chord
import pretty_midi
import plotly.express as px
import plotly.graph_objects as go

from utils import load_pickle
from utils import preprocess_midi
from utils import get_midi_statistics
from utils import predict
from utils import create_report

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Music Genre Classification",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>

.main{
    background:#0E1117;
}

.stMetric{
    background:#1b1d24;
    padding:15px;
    border-radius:12px;
}

div[data-testid="stMetric"]{
    background:#1b1d24;
    border-radius:12px;
    padding:15px;
}

.block-container{
    padding-top:2rem;
}

h1,h2,h3,h4{
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL
# ======================================================

@st.cache_resource
def load_model(model_name):

    return tf.keras.models.load_model(model_name)


# ======================================================
# LOAD PICKLES
# ======================================================

@st.cache_resource
def load_pickle(file):

    with open(file,"rb") as f:
        return pickle.load(f)


note_to_int = load_pickle("models/note_to_int.pkl")
label_encoder = load_pickle("models/label_encoder.pkl")


# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("🎵 Music Genre Classification")

model_choice = st.sidebar.radio(

    "Choose Model",

    [
        "Original LSTM",
        "Tuned LSTM"
    ]

)

if model_choice=="Original LSTM":

    model = load_model("models/best_lstm.keras")

else:

    model = load_model("models/best_lstm_tuned.keras")

st.sidebar.markdown("---")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.markdown(
"""
### Supported Formats

- MID
- MIDI

Upload a MIDI file and the model will classify
its genre.
"""
)

# ======================================================
# TITLE
# ======================================================

st.title("🎵 Music Genre Classification using Deep Learning")

st.write(
"""
Upload a **MIDI (.mid/.midi)** file and let the trained
Bidirectional LSTM predict its music genre.
"""
)

# ======================================================
# FILE UPLOADER
# ======================================================

uploaded_file = st.file_uploader(

    "Upload MIDI File",

    type=["mid","midi"]

)

# ======================================================
# MIDI EXTRACTION
# ======================================================

MAX_SEQUENCE_LENGTH = 500


def extract_notes(midi_path):

    notes=[]

    midi=converter.parse(midi_path)

    parts=instrument.partitionByInstrument(midi)

    if parts:

        notes_to_parse=parts.parts[0].recurse()

    else:

        notes_to_parse=midi.flat.notes

    for element in notes_to_parse:

        if isinstance(element,note.Note):

            notes.append(str(element.pitch))

        elif isinstance(element,chord.Chord):

            notes.append(
                ".".join(str(n) for n in element.normalOrder)
            )

    return notes[:MAX_SEQUENCE_LENGTH]


# ======================================================
# ENCODE NOTES
# ======================================================

def preprocess(notes):

    encoded=[]

    for n in notes:

        encoded.append(

            note_to_int.get(n,0)

        )

    encoded=pad_sequences(

        [encoded],

        maxlen=MAX_SEQUENCE_LENGTH,

        padding="post"

    )

    return encoded


# ======================================================
# MIDI INFORMATION
# ======================================================

def midi_information(file_path):

    pm=pretty_midi.PrettyMIDI(file_path)

    tempo=pm.estimate_tempo()

    duration=pm.get_end_time()

    note_count=0

    pitches=[]

    for ins in pm.instruments:

        note_count+=len(ins.notes)

        for n in ins.notes:

            pitches.append(n.pitch)

    return {

        "tempo":tempo,

        "duration":duration,

        "note_count":note_count,

        "pitches":pitches

    }

# ======================================================
# WAIT FOR FILE
# ======================================================

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False,suffix=".mid") as tmp:

        tmp.write(uploaded_file.read())

        temp_path=tmp.name

    st.success("MIDI file uploaded successfully.")

    with st.spinner("Analyzing MIDI file..."):

        notes = extract_notes(temp_path)

        sequence = preprocess(notes)

        midi_stats = midi_information(temp_path)
    midi_stats = get_midi_statistics(temp_path)

    prediction = predict(
        model,
        temp_path,
        note_to_int,
        label_encoder
    )

    predicted_genre = prediction["genre"]

    confidence = prediction["confidence"]

    probabilities = prediction["probabilities"]
    # ======================================================
    # PREDICTION
    # ======================================================

    prediction = model.predict(sequence, verbose=0)

    probabilities = prediction[0]

    predicted_index = np.argmax(probabilities)

    predicted_genre = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = probabilities[predicted_index] * 100

    # ======================================================
    # DISPLAY PREDICTION
    # ======================================================

    st.markdown("---")

    st.subheader("Prediction")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Genre",
            predicted_genre
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    # ======================================================
    # PROBABILITY CHART
    # ======================================================

    genres = label_encoder.classes_

    probability_df = pd.DataFrame({
        "Genre": genres,
        "Probability": probabilities * 100
    })

    fig = px.bar(
        probability_df,
        x="Probability",
        y="Genre",
        orientation="h",
        text="Probability",
        color="Probability",
        color_continuous_scale="Viridis"
    )

    fig.update_layout(
        title="Prediction Probabilities",
        height=400,
        showlegend=False
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ======================================================
    # MIDI INFORMATION
    # ======================================================

    st.markdown("---")

    st.subheader("MIDI Statistics")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Tempo",
        f"{midi_stats['tempo']:.2f} BPM"
    )

    c2.metric(
        "Duration",
        f"{midi_stats['duration']:.2f} sec"
    )

    c3.metric(
        "Total Notes",
        midi_stats["total_notes"]
    )

    # ======================================================
    # PITCH HISTOGRAM
    # ======================================================

    st.markdown("---")

    st.subheader("Pitch Distribution")

    if len(midi_stats["pitches"]) > 0:

        hist = px.histogram(
            x=midi_stats["pitches"],
            nbins=40,
            labels={
                "x": "MIDI Pitch",
                "y": "Count"
            },
            title="Distribution of MIDI Note Pitches"
        )

        st.plotly_chart(
            hist,
            use_container_width=True
        )

    # ======================================================
    # CONFIDENCE GAUGE
    # ======================================================

    st.markdown("---")

    st.subheader("Prediction Confidence")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={
                "text":"Confidence (%)"
            },
            gauge={
                "axis":{"range":[0,100]},
                "bar":{"color":"green"},
                "steps":[
                    {"range":[0,50],"color":"lightgray"},
                    {"range":[50,75],"color":"yellow"},
                    {"range":[75,100],"color":"lightgreen"}
                ]
            }
        )
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    # ======================================================
    # REPORT
    # ======================================================

    report = f"""
Music Genre Classification Report

File:
{uploaded_file.name}

Predicted Genre:
{predicted_genre}

Confidence:
{confidence:.2f} %

Tempo:
{midi_stats['tempo']:.2f} BPM

Duration:
{midi_stats['duration']:.2f} seconds

Total Notes:
{midi_stats['total_notes']}
"""

    st.download_button(
        "Download Prediction Report",
        report,
        file_name="prediction_report.txt"
    )

    # ======================================================
    # RAW PROBABILITIES
    # ======================================================

    with st.expander("View Prediction Probabilities"):

        st.dataframe(
            probability_df.sort_values(
                "Probability",
                ascending=False
            ),
            use_container_width=True
        )

    os.remove(temp_path)

# ======================================================
# NO FILE
# ======================================================

else:

    st.info(
        "Please upload a MIDI (.mid/.midi) file to begin."
    )

# ======================================================
# FOOTER
# ======================================================

st.markdown("---")

st.caption(
    "Music Genre Classification using Bidirectional LSTM • Built with Streamlit and TensorFlow"
)