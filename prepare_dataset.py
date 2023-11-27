import musdb
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from multiprocessing import cpu_count
import soundfile as sf
from config import *
import librosa
import sys


def convert_to_wav(audio, subset_dir, track_name, filename):
    output_path = f"./musdb18_extracted/{subset_dir}/{track_name}"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    sf.write(f"{output_path}/{filename}.wav", audio, ORIG_SR)
    print(f"Saving {output_path}/{filename}.wav")


def resample_wav(subset_dir, track_name, filename):
    input_path = f"./musdb18_extracted/{subset_dir}/{track_name}/{filename}.wav"
    output_path = f"./musdb18_resized/{subset_dir}/{track_name}"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    wav, sr = librosa.load(input_path, sr=SR)
    sf.write(f"{output_path}/{filename}.wav", wav, SR)
    print(f"Saving {output_path}/{filename}.wav")


def extract_and_resample_files(subset):
    musdb18 = musdb.DB(root='./musdb18', subsets=[subset])

    subset_dir = "Dev"
    if subset == "test":
        subset_dir = "Test"

    with ThreadPoolExecutor(max_workers=cpu_count() * 2) as pool:
        for track in musdb18.tracks:
            pool.submit(convert_to_wav, track.audio, subset_dir, track.name, "mix")
            pool.submit(convert_to_wav, track.targets['vocals'].audio, subset_dir, track.name,
                        "vocals")
            pool.submit(convert_to_wav, track.targets['accompaniment'].audio, subset_dir, track.name,
                        "accompaniment")

    print(f"Extraction completed successfully for {subset_dir} files")

    with ThreadPoolExecutor(max_workers=cpu_count() * 2) as pool:
        for track in musdb18.tracks:
            pool.submit(resample_wav, subset_dir, track.name, "mix")
            pool.submit(resample_wav, subset_dir, track.name, "vocals")
            pool.submit(resample_wav, subset_dir, track.name, "accompaniment")

    print(f"Resampling completed successfully for {subset_dir} files")


if __name__ == "__main__":
    args = sys.argv
    if len(args) == 1:
        print("Usage: prepare_dataset.py <subset>")
        exit(-1)
    subset = args[1].lower()
    if subset != "train" and subset != "test":
        print("Invalid argument. subset must be 'train' or 'test'")
    extract_and_resample_files(subset)
