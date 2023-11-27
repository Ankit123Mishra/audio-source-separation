import numpy as np
from config import *
from model import unet
from librosa.util import find_files
import sys


def load_npz(target=None, first=None):
    npz_files = find_files('./musdb18_npz', ext="npz")[:first]
    for file in npz_files:
        npz = np.load(file)
        assert(npz["mix"].shape == npz[target].shape)
        yield npz['mix'], npz[target]


def sample_patches(mix_mag, target_mag):
    X, y = [], []
    for mix, target in zip(mix_mag, target_mag):
        starts = np.random.randint(0, mix.shape[1] - PATCH_SIZE, (mix.shape[1] - PATCH_SIZE) // SAMPLE_STRIDE)
        for start in starts:
            end = start + PATCH_SIZE
            X.append(mix[1:, start:end, np.newaxis])
            y.append(target[1:, start:end, np.newaxis])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        print("Usage: train.py <target>")
        exit(-1)
    target = args[1].lower()
    if target != "vocals" and target != "accompaniment":
        print("Invalid argument. Target must be 'vocals' or 'accompaniment'")
        exit(-1)
    mix_mag, target_mag = zip(*load_npz(target=target, first=-1))

    model = unet()
    model.compile(optimizer='adam', loss='mean_absolute_error')

    for e in range(EPOCH):
        X, y = sample_patches(mix_mag, target_mag)
        model.fit(X, y, batch_size=BATCH, verbose=1, validation_split=0.01)
        model.save('../models/{target}_{num}.h5'.format(target=target, num=e+1), overwrite=True)
