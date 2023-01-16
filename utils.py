import collections
import datetime

import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf

from IPython import display
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple
# filenames = glob.glob(str('/content/bach/**/*.mid*'))
# print('Number of files:', len(filenames))
# sample_file = filenames[1]
# print(sample_file)

# # Sampling rate for audio playback
# pm = pretty_midi.PrettyMIDI(sample_file)

_SAMPLING_RATE = 16000
def display_audio(pm: pretty_midi.PrettyMIDI, seconds=30):
  waveform = pm.fluidsynth(fs=_SAMPLING_RATE)
  # Take a sample of the generated waveform to mitigate kernel resets
  waveform_short = waveform[:seconds*_SAMPLING_RATE]
  return display.Audio(waveform_short, rate=_SAMPLING_RATE)
display_audio(pm)

# print('Number of instruments:', len(pm.instruments))
# instrument = pm.instruments[0]
# instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
# print('Instrument name:', instrument_name)

for i in range(len(pm.instruments)):
    instrument_name = pretty_midi.program_to_instrument_name(pm.instruments[i].program)
    print(instrument_name)

def return_instruments(pm: pretty_midi.PrettyMIDI):
  list_instr = []
  for i in range(len(pm.instruments)):
    instrument_name = pretty_midi.program_to_instrument_name(pm.instruments[i].program)
    list_instr.append(instrument_name)
  return set(list_instr)

pm = pretty_midi.PrettyMIDI(filenames[1])
print(f'There are {len(return_instruments(pm))} instruments:\n{return_instruments(pm)}')

pm.instruments[0]

def midi_to_notes_single_inst(midi_file: str) -> pd.DataFrame:
  # Load the MIDI file into a PrettyMIDI object
  midi_data = pretty_midi.PrettyMIDI(midi_file)
  pitch_to_note = np.vectorize(pretty_midi.note_number_to_name)
  # Extract the notes from the first instrument
  notes = midi_data.instruments[0].notes
    # Create a list to store the steps
  steps = []
  # Set the initial step to be the start time of the first note
  prev_start = notes[0].start
  for note in notes:
    start = note.start
    # Calculate the step as the difference between the start time of the current note and the start time of the previous note
    step = start - prev_start
    steps.append(step)
    prev_start = start
  # Create a DataFrame with one row for each note
  df = pd.DataFrame(
      {'pitch': [note.pitch for note in notes],
       'start': [note.start for note in notes],
       'end': [note.end for note in notes],
       'step':steps,
       'duration': [note.end - note.start for note in notes],
       "note": [pitch_to_note(note.pitch) for note in notes]})
  return df, notes


def midi_to_notes_all_inst(midi_file: str) -> pd.DataFrame:
  # Load the MIDI file into a PrettyMIDI object
  midi_data = pretty_midi.PrettyMIDI(midi_file)
  pitch_to_note = np.vectorize(pretty_midi.note_number_to_name)

  # Create a list to store the notes from each instrument
  all_notes = []
  
  # Iterate over the instruments
  for i, instrument in enumerate(midi_data.instruments):
    # Extract the notes from the instrument
    notes = instrument.notes
    # Create a list to store the steps
    steps = []
    # Set the initial step to be the start time of the first note
    prev_start = notes[0].start
    for note in notes:
      start = note.start
      # Calculate the step as the difference between the start time of the current note and the start time of the previous note
      step = start - prev_start
      steps.append(step)
      prev_start = start
    # Create a DataFrame with one row for each note
    df = pd.DataFrame(
        {'pitch': [note.pitch for note in notes],
         'start': [note.start for note in notes],
         'end': [note.end for note in notes],
         'step':steps,
         'duration': [note.end - note.start for note in notes],
         'instrument': [pretty_midi.program_to_instrument_name(midi_data.instruments[i].program) for _ in notes], 
         'note':[pitch_to_note(note.pitch) for note in notes]})

    # Add the notes from this instrument to the list
    all_notes.append(df)

  # Concatenate the notes from all the instruments into a single DataFrame
  df = pd.concat(all_notes)

  return df, notes

def plot_inst_roll(notes: pd.DataFrame, num: Optional[int] = None):
  from itertools import cycle
  if num:
    title = f'First {num} notes'
  else:
    title = f'Whole track'
    count = len(notes['pitch'])
  plt.figure(figsize=(20, 4))
  # Get the current color cycle
  colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
  color_iter = cycle(colors)
  # Iterate over the instruments
  for i, (instrument, group) in enumerate(notes.groupby('instrument')):
    plot_pitch = np.stack([group['pitch'], group['pitch']], axis=0)
    plot_time = np.stack([group['start'], group['end']], axis=0)
    # Use the next color in the color cycle for the current instrument
    color = next(color_iter)
    plt.plot(plot_time[:, :num], plot_pitch[:, :num], color=color, marker=".")
  plt.xlabel('Time [s]')
  plt.ylabel('Pitch')
  _ = plt.title(title)

def dataframe_to_tensor(df, R, S, instruments):
    """
    Converts a dataframe with columns 'pitch', 'start', 'end', 'duration', and 'instrument' into a tensor x ∈ {0, 1}R×S×M,
    where R is the time, S is the note, and M is the instrument.
    
    Parameters:
        df (pd.DataFrame): Dataframe with columns 'pitch', 'start', 'end', 'duration', and 'instrument'.
        R (int): Number of time steps in a bar.
        S (int): Number of note candidates.
        instruments (list): List of unique instruments in the dataframe.
    
    Returns:
        x (np.ndarray): Tensor x ∈ {0, 1}R×S×M.
    """
    # Create empty tensor
    x = np.zeros((R, S, len(instruments)))
    
    # Iterate through dataframe
    for i, row in df.iterrows():
        pitch = row['pitch']
        start = row['start']
        end = row['end']
        duration = row['duration']
        instrument = row['instrument']
        
        # Calculate time step and note candidate indices
        start_idx = int(start // R)
        end_idx = int(end // R)
        note_idx = pitch % S
        
        # Set values in tensor
        x[start_idx:end_idx+1, note_idx, instruments.index(instrument)] = 1
    
    return x

# dd = midi_to_notes_all_inst(filenames[1])[0]
# # dd['instrument'] = dd['instrument'].apply(lambda x: np.random.choice(['Piano', 'Guitar']))
# # Determine values of R, S, and instruments
# R = dd.shape[0]
# S = 12
# instruments = list(dd['instrument'].unique())

# # Convert dataframe to tensor
# x = dataframe_to_tensor(dd, R, S, instruments)
# x.shape

# plot_inst_roll(dd, 100)

# midi_to_notes_all_inst(filenames[1])[0]

def notes_to_midi(
  notes: pd.DataFrame,
  out_file: str, 
  instrument_name: str
) -> pretty_midi.PrettyMIDI:

  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(
      program=pretty_midi.instrument_name_to_program(
          instrument_name))

  prev_start = 0
  for i, note in notes.iterrows():
    start = float(prev_start + note['step'])
    end = float(start + note['duration'])
    note = pretty_midi.Note(
        velocity = np.random.randint(0,127), #lol just to be chaotic
        pitch=int(note['pitch']),
        start=start,
        end=end,
    )
    instrument.notes.append(note)
    prev_start = start

  pm.instruments.append(instrument)
  pm.write(out_file)
  return pm
