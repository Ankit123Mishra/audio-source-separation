import numpy as np
from librosa.core import istft, load, stft, magphase
from config import *
import keras as keras
import soundfile as sf
import sys
from pathlib import Path


def predict(target, mix_file, save_files=True):
    Path("./results").mkdir(parents=True, exist_ok=True)
    target_model_name = "vocals_20"
    if target == "accompaniment":
        target_model_name = "accompaniment_20"

    mix_wav, _ = load(mix_file, sr=SR)
    mix_wav_mag, mix_wav_phase = magphase(stft(mix_wav, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH))

    START = 0
    END = START + 128

    mix_wav_mag = mix_wav_mag[:, START:END]
    mix_wav_phase = mix_wav_phase[:, START:END]

    model = keras.models.load_model(f'./models/{target_model_name}.h5')
    X = mix_wav_mag[1:].reshape(1, 512, 128, 1)
    y = model.predict(X, batch_size=32)

    target_pred_mag = np.vstack((np.zeros((128)), y.reshape(512, 128)))

    target_pred_waveform = istft(
        target_pred_mag * mix_wav_phase
        , win_length=WINDOW_SIZE, hop_length=HOP_LENGTH)
    mix_waveform = istft(
        mix_wav_mag * mix_wav_phase
        , win_length=WINDOW_SIZE, hop_length=HOP_LENGTH)

    if save_files:
        sf.write(f'./results/{target}_sample_pred.wav', target_pred_waveform, SR)
        sf.write(f'./results/mix_sample_downsampled.wav', mix_waveform, SR)

    return mix_waveform, target_pred_waveform


def predict_full(target, mix_file, save_files=True):
    Path("./results").mkdir(parents=True, exist_ok=True)
    # load test audio and convert to mag/phase
    target_model_name = "vocals_20"
    if target == "accompaniment":
        target_model_name = "accompaniment_20"

    mix_wav, _ = load(mix_file, sr=SR)
    mix_wav_mag, mix_wav_phase = magphase(stft(mix_wav, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH))

    START = 0
    END = START + 128

    mix_complete_wav_mag, mix_complete_wav_phase = mix_wav_mag, mix_wav_phase

    target_pred_mag = None
    for s in range(0, mix_complete_wav_mag.shape[1], 128):
        diff = mix_complete_wav_mag.shape[1] - s
        if (s + 128) > mix_complete_wav_mag.shape[1]:
            END = mix_complete_wav_mag.shape[1]
        else:
            END = s + 128
        mix_wav_mag = mix_complete_wav_mag[:, s:END]

        # load saved model
        model = keras.models.load_model(f'./models/{target_model_name}.h5')

        # predict each chunk and append to the target tensor
        if mix_wav_mag.shape[1] < 128:
            diff_shape = 128 - diff
            mix_wav_mag = np.hstack((mix_wav_mag, np.zeros((513, diff_shape))))
        X = mix_wav_mag[1:].reshape(1, 512, 128, 1)
        y = model.predict(X, batch_size=32)

        if target_pred_mag is None:
            target_pred_mag = np.vstack((np.zeros((128)), y.reshape(512, 128)))
        else:
            target_pred_mag = np.hstack((target_pred_mag, np.vstack((np.zeros((128)), y.reshape(512, 128)))))

    target_pred_mag = target_pred_mag[:, :mix_complete_wav_mag.shape[1]]

    target_pred_waveform = istft(
        target_pred_mag * mix_complete_wav_phase
        , win_length=WINDOW_SIZE, hop_length=HOP_LENGTH)

    mix_waveform = istft(
        mix_complete_wav_mag * mix_complete_wav_phase
        , win_length=WINDOW_SIZE, hop_length=HOP_LENGTH)

    if save_files:
        sf.write(f'./results/{target}_complete_pred.wav', target_pred_waveform, SR)
        sf.write(f'./results/mix_complete_downsampled.wav', mix_waveform, SR)

    return mix_waveform, target_pred_waveform


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        print("Usage: predict.py <target>")
        exit(-1)
    target = args[1].lower()
    if target != "vocals" and target != "accompaniment":
        print("Invalid argument. Target must be 'vocals' or 'accompaniment'")
        exit(-1)
    if len(args) == 3 and args[2].lower() == "--full":
        predict_full(target, "mix.wav")
    else:
        predict(target, "mix.wav")

