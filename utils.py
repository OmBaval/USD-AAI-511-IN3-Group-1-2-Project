import pickle
import pretty_midi

from music21 import converter
from music21 import instrument
from music21 import note
from music21 import chord

from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 500


def load_pickle(path):
    """
    Load a pickle file.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_notes(midi_path):
    """
    Extract notes and chords from a MIDI file.

    Notes:
        C4
        E5

    Chords:
        0.4.7
    """

    notes = []

    midi = converter.parse(midi_path)

    parts = instrument.partitionByInstrument(midi)

    if parts:

        notes_to_parse = parts.parts[0].recurse()

    else:

        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:

        if isinstance(element, note.Note):

            notes.append(str(element.pitch))

        elif isinstance(element, chord.Chord):

            notes.append(
                ".".join(str(n) for n in element.normalOrder)
            )

    return notes[:MAX_SEQUENCE_LENGTH]


def encode_notes(notes, note_to_int):
    """
    Convert notes into integer sequence.
    """

    encoded = []

    for n in notes:

        encoded.append(
            note_to_int.get(n, 0)
        )

    return encoded


def preprocess_midi(midi_path, note_to_int):
    """
    Complete preprocessing pipeline.

    MIDI
        ↓
    Extract Notes
        ↓
    Encode
        ↓
    Pad Sequence
        ↓
    Ready for LSTM
    """

    notes = extract_notes(midi_path)

    encoded = encode_notes(notes, note_to_int)

    sequence = pad_sequences(
        [encoded],
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post"
    )

    return sequence, notes


def get_midi_statistics(midi_path):
    """
    Compute useful MIDI statistics.
    """

    pm = pretty_midi.PrettyMIDI(midi_path)

    tempo = pm.estimate_tempo()

    duration = pm.get_end_time()

    total_notes = 0

    pitches = []

    instruments = len(pm.instruments)

    for inst in pm.instruments:

        total_notes += len(inst.notes)

        for n in inst.notes:

            pitches.append(n.pitch)

    unique_notes = len(set(pitches))

    return {
    "tempo": tempo,
    "duration": duration,
    "total_notes": total_notes,   # instead of total_notes
    "unique_notes": unique_notes,
    "instrument_count": instruments,
    "pitches": pitches
    }


def predict(model,
            midi_path,
            note_to_int,
            label_encoder):
    """
    Perform complete prediction.
    """

    sequence, notes = preprocess_midi(
        midi_path,
        note_to_int
    )

    prediction = model.predict(
        sequence,
        verbose=0
    )

    probabilities = prediction[0]

    predicted_index = probabilities.argmax()

    genre = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = float(
        probabilities[predicted_index]
    ) * 100

    return {

        "genre": genre,

        "confidence": confidence,

        "probabilities": probabilities,

        "notes": notes,

        "sequence": sequence

    }


def create_report(filename,
                  prediction,
                  stats):
    """
    Create downloadable report.
    """

    report = f"""
Music Genre Classification Report
=================================

Filename:
{filename}

Predicted Genre:
{prediction['genre']}

Confidence:
{prediction['confidence']:.2f}%

Tempo:
{stats['tempo']:.2f} BPM

Duration:
{stats['duration']:.2f} seconds

Total Notes:
{stats['total_notes']}

Unique Notes:
{stats['unique_notes']}

Instruments:
{stats['instrument_count']}

Generated using:
Bidirectional LSTM Music Genre Classifier
"""

    return report