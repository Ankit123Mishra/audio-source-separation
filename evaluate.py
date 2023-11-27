import os
import numpy as np
import museval
from predict import predict
import sys


def evaluate(target):
    input_path = f"./musdb18_extracted/Test"
    sdr = []
    sar = []
    for root, dirs, files in os.walk(input_path, topdown=False):
        for name in dirs:
            mix_wav, target_wav = predict(target, f"{root}/{name}/mix.wav", save_files=False)
            try:
                scores = museval.metrics.bss_eval(mix_wav, target_wav)
                sdr.append(scores[0])
                sar.append(scores[3])
            except:
                continue

    return np.mean(sdr), np.mean(sar)


if __name__ == "__main__":
    args = sys.argv
    if len(args) == 1:
        print("Usage: train.py <target>")
        exit(-1)
    target = args[1].lower()
    if target != "vocals" and target != "accompaniment":
        print("Invalid argument. Target must be 'vocals' or 'accompaniment'")
        exit(-1)
    print(f"Evaluating metrics for {target}...")
    mean_sdr, mean_sar = evaluate(target)
    print(f"\nMean SDR: {mean_sdr}\nMean SAR: {mean_sar}")