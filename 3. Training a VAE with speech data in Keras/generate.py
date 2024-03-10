import os
import pickle

import numpy as np
import soundfile as sf

from soundgenerator import SoundGenerator
from vae import VAE

from train import SPECTROGRAMS_PATH

HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = r"D:\personal projects\Music Generation\samples\SAVE_DIR_ORIGINAL"
SAVE_DIR_GENERATED = r"D:\personal projects\Music Generation\samples\SAVE_DIR_GENERATED"
MIN_MAX_VALUES_PATH = r"D:\personal projects\Music Generation\samples\MIN_MAX_VALUES_SAVE_DIR\min_max_values.pkl"


def load_fsdd(spectrograms_path):
    x_train = [] #an empty list to be filled with spectrograms.
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) # now, had the spectrogram.
            # (n_bins which from stft, n_frames, 1) #1 is the extra dimension.
            x_train.append(spectrogram) # stored in x_train list.
            file_paths.append(file_path)
    x_train = np.array(x_train) # cast to numpy array.
    x_train = x_train[..., np.newaxis] # np.newaxis is used to increase the dimensionality of an array by one unit. # -> (3000 # of data we have in dataset, 256 num bins, 64 frames, 1) # treat spectrograms as grey scale images.
    return x_train, file_paths

# function to SAMPLE a few spectrograms.
def select_spectrograms(spectrograms,
                        file_paths,
                        min_max_values,
                        num_spectrograms = 2):
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrograms = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max_values = [min_max_values[file_path] for file_path in file_paths]

    print(file_paths)
    print(sampled_min_max_values)
    return sampled_spectrograms, sampled_min_max_values

def save_signals(signals, save_dir, sample_rate = 22050):
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate) #write into the sound file sf.



if __name__ == "__main__":
    # initialise sound generator
    vae = VAE.load("model") # in model file
    sound_generator = SoundGenerator(vae, HOP_LENGTH)

    # load spectrograms + min_max_values
    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f) # pickle.load(f) is a Python function used to deserialize a binary file containing a Python object serialized with the pickle module
    specs, file_paths = load_fsdd(SPECTROGRAMS_PATH)

    # sample a few spectrograms + a few min_max_values
    sampled_specs, sampled_min_max_values = select_spectrograms(specs,
                                                                file_paths,
                                                                min_max_values,
                                                                5) # sample '5' spectrograms.

    # gen audio from sampled spectrograms!
    signals, _ = sound_generator.generate(sampled_specs, sampled_min_max_values)

    # direct converting spectrogram samples to audio (for comparison)
    original_signals = sound_generator.convert_spectrograms_to_audio(sampled_specs, sampled_min_max_values)

    # save audio signals.
    save_signals(signals, SAVE_DIR_GENERATED) #saved files.
    save_signals(original_signals, SAVE_DIR_ORIGINAL)