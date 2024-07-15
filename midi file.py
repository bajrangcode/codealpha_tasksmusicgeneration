from music21 import stream, note, chord, instrument

# Create a stream
output_stream = stream.Stream()

# Add an instrument (optional)
output_stream.append(instrument.Piano())

# Add notes and chords
notes = [
    note.Note('C4', quarterLength=1.0),
    note.Note('E4', quarterLength=1.0),
    note.Note('G4', quarterLength=1.0),
    chord.Chord(['C4', 'E4', 'G4'], quarterLength=1.0),
    note.Rest(quarterLength=1.0),  # Adding a rest
    note.Note('D4', quarterLength=1.0),
    chord.Chord(['F4', 'A4', 'C5'], quarterLength=2.0)
]

# Append notes to the stream
for element in notes:
    output_stream.append(element)

# Save the stream as a MIDI file
output_stream.write('midi', fp='complex_output.mid')
