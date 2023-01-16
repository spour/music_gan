gan = define_gan(gen, disc)

def generate_real_samples(dataset, n_samples):
    ix = np.random.randint(0, 1, n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, 1))
    return X, y
 
def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input
    
def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = g_model.predict(x_input)
    y = zeros((n_samples, 1))
    return X, y

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=51, n_batch=10):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            d_loss, _ = d_model.train_on_batch(X, y)
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
        if (i+1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)
            clear_output()

X_train =[]

for x in range(10):
  dd = midi_to_notes_all_inst(filenames[x])[0]
  # dd['instrument'] = dd['instrument'].apply(lambda x: np.random.choice(['Piano', 'Guitar']))
  # Determine values of R, S, and instruments
  R = dd.shape[0]
  S = 84
  instruments = list(dd['instrument'].unique())
  X_train.append(dataframe_to_tensor(dd, R, S, instruments))

X_train = [d[:100,:,:] for d in X_train]

test = tf.stack(X_train)

test

np.random.choice([1, 0], size=10)
test.shape
np.unique(np.squeeze(test[:1]), return_counts = True)

X_train

disc.train_on_batch(test, np.random.choice([1, 0], size=10))

gan.train_on_batch(test, np.random.choice([1, 0], size=10))

def generate_latent_points(latent_dim, shape):
    """Generate latent points with a given shape and dimension.
    
    Arguments:
        latent_dim: int, the latent dimension of the points.
        shape: tuple, the shape of the output tensor.
    
    Returns:
        latent_points: array, the latent points with the given shape and dimension.
    """
    # Generate random normal samples with shape (shape[0], shape[1], ..., latent_dim)
    latent_points = np.random.randn(*shape, latent_dim)
    return latent_points
latent_dim = 100
shape = (16, 100, 84, 1)

latent_points = generate_latent_points(latent_dim, shape)
# X = gan.predict(latent_points)

np.squeeze(latent_points, 3)[:,:,:,:1].shape

preds = gen.predict_on_batch(np.squeeze(latent_points, 3)[:,:,:,:1])

# binarize the output
output = np.squeeze(preds[1])
binarized_output = np.where(output > 0, 1, 0)

# Create a piano roll matrix
pr = binarized_output.T

# Create a PrettyMIDI object from the piano roll
pm = pretty_midi.pretty_midi.PrettyMIDI()
inst = pretty_midi.Instrument(program=0)
pm.instruments.append(inst)
inst.piano_roll = pr
pr *=100

np.squeeze(np.where(preds[1] > 0, 1, 0)[:10])

def matrix_to_midi(matrix, program, file_name):
    # Create a PrettyMIDI object
    midi_object = pretty_midi.PrettyMIDI()
    # Create an Instrument instance
    program = pretty_midi.instrument_name_to_program('Cello')
    instrument = pretty_midi.Instrument(program=program)
    # Get the number of rows and columns in the matrix
    rows, cols = matrix.shape
    # Initialize the current time to 0
    current_time = 0
    # Iterate over the rows and columns of the matrix
    for row in range(rows):
        for col in range(cols):
            # Check if the entry at this position is non-zero
            if matrix[row, col] > 0:
                # Create a Note instance with the pitch, start time, end time and velocity
                note = pretty_midi.Note(
                    velocity=matrix[row, col], pitch=row, start=current_time, end=current_time+0.01)
                current_time = current_time + 0.01
                # Add the note to the instrument
                instrument.notes.append(note)
    # Add the instrument to the PrettyMIDI object
    midi_object.instruments.append(instrument)
    # Write out the MIDI data
    midi_object.write(file_name)


# Example usage
matrix = np.random.randint(low=50, high=80, size=(128, 500))
matrix_to_midi(pr.T, program="Cello", file_name='generated_music.mid')

pr

midi_to_notes_single_inst('/content/generated_music.mid')

display_audio(pretty_midi.PrettyMIDI("/content/generated_music.mid"))

def tensor_to_dataframe(x, R, S, instruments):
  """
  Converts a tensor x ∈ {0, 1}R×S×M, where R is the time, S is the note, and M is the instrument
  into a dataframe with columns 'pitch', 'start', 'end', 'duration', and 'instrument'.
  
  Parameters:
      x (np.ndarray): Tensor x ∈ {0, 1}R×S×M.
      R (int): Number of time steps in a bar.
      S (int): Number of note candidates.
      instruments (list): List of unique instruments in the dataframe.
  
  Returns:
      df (pd.DataFrame): Dataframe with columns 'pitch', 'start', 'end', 'duration', and 'instrument'.
  """
  # Initialize empty lists
  pitches = []
  starts = []
  ends = []
  durations = []
  instrs = []

  # Iterate through tensor
  for i in range(x.shape[0]):
      for j in range(x.shape[1]):
          for k in range(x.shape[2]):
              if x[i, j, k] == 1: 
                  pitch = j + S * k
                  start = i * R
                  end = (i+1) * R
                  duration = end - start
                  inst = instruments[k]
                  pitches.append(pitch)
                  starts.append(start)
                  ends.append(end)
                  durations.append(duration)
                  instrs.append(inst)
  # Create the dataframe and return
  df = pd.DataFrame({'pitch': pitches, 'start': starts, 'end': ends, 'duration': durations, 'instrument': instrs})
  return df

tensor_to_dataframe(np.where(preds[1]>0, 1, 0), *(100, 84, ["Acoustic Grand Piano"])).head(30)
