import musdb
import librosa
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

from config import *
import numpy as np


def extract_magnitude_spectrogram(audio):
    spectrogram = librosa.stft(audio, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH)
    mag, _ = librosa.magphase(spectrogram)
    return mag.astype(np.float32)


def save_to_npz(mix, vocals, accompaniment, filename):
    mix_mag_spec = extract_magnitude_spectrogram(mix)
    vocals_mag_spec = extract_magnitude_spectrogram(vocals)
    accompaniment_mag_spec = extract_magnitude_spectrogram(accompaniment)

    norm = mix_mag_spec.max()
    mix_mag_spec /= norm
    vocals_mag_spec /= norm
    accompaniment_mag_spec /= norm

    output_path = "./musdb18_npz"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(os.path.join(output_path, filename + '.npz'), mix=mix_mag_spec, vocals=vocals_mag_spec,
                        accompaniment=accompaniment_mag_spec)
    print(f"Saved file {output_path}/{filename}.npz successfully!")


def preprocess():
    musdb18_train = musdb.DB(root='../musdb18', subsets=["train"])
    input_path = f"./musdb18_resized/Dev"

    with ThreadPoolExecutor(max_workers=cpu_count() * 2) as pool:
        for track in musdb18_train.tracks:
            mix, _ = librosa.load(f"{input_path}/{track.name}/mix.wav", sr=SR)
            vocals, _ = librosa.load(f"{input_path}/{track.name}/vocals.wav", sr=SR)
            accompaniment, _ = librosa.load(f"{input_path}/{track.name}/accompaniment.wav", sr=SR)
            pool.submit(save_to_npz, mix, vocals, accompaniment, track.name)


if __name__ == "__main__":
    preprocess()
