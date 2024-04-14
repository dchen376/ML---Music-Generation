from preprocess import MinMaxNormaliser

import librosa

class SoundGenerator:
    """SoundGenerator is responsible for generating audios from
    spectrograms.
    """

    def __init__(self, vae, hop_length):
        self.vae = vae
        self.hop_length = hop_length
        self._min_max_normaliser = MinMaxNormaliser(0, 1)

    def generate(self, spectrograms, min_max_values):
        generated_spectrograms, latent_representations = self.vae.reconstruct(spectrograms)

        signals = self.convert_spectrograms_to_audio(generated_spectrograms, min_max_values)
        return signals, latent_representations

    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
        signals = [] # list where we add converted spectrograms.

        # The zip() function in Python is used to combine multiple iterables
        # (such as lists, tuples, etc.) element-wise. It takes iterables as input
        # and returns an iterator of tuples where each tuple contains the elements
        # from the input iterables at the same index.
        for spectrogram, min_max_value in zip(spectrograms, min_max_values):

            # reshape the log spectrogram (which is the 'spectrogram' in prev line)
            log_spectrogram = spectrogram[:, :, 0] # copy all 1st dimension, copy all 2nd dimension, drop the 3rd dimension.

            # apply denormalization
            denorm_log_spec = self._min_max_normaliser.denormalise(
                log_spectrogram, min_max_value["min"], min_max_value["max"])

            # log spectrogram -> (linear) spectrogram
            linear_spect = librosa.db_to_amplitude(denorm_log_spec)

            # finally, apply Griffin-Lim
            signal = librosa.istft(linear_spect, hop_length = self.hop_length)

            # append signal to "signals" list we had above.
            signals.append(signal)

        return signals
