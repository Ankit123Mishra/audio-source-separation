## Audio Source Separation

Source Separation is the process of isolating individual sounds in an auditory mixture of multiple sounds. In this project, we try to train deep models which can separate vocals and instrumental (accompaniment) sources from a mix audio signal.

![Music Source Separation](https://source-separation.github.io/tutorial/_images/source_separation_io.png)

### Dataset:
We use the publicly available MUSDB18 dataset.

![Dataset](https://sigsep.github.io/assets/img/musheader.41c6bf29.png)

The musdb18 is a dataset of 150 full lengths music tracks (~10h duration) of different genres along with their isolated drums, bass, vocals and others stems.
musdb18 contains two folders, a folder with a training set: "train", composed of 100 songs, and a folder with a test set: "test", composed of 50 songs.

### Parsing MUSDB18 Dataset:
We use the **musdb** library for this task.

```console
    python3 prepare_dataset.py train
```
The above command assumes that you have "musb18" directory already extracted in the current directory.
This will prepare the wav files and resample audio files for preprocessing.

### Preprocessing:
The below command will extract magnitude spectrogram from the train dataset and compress and save it as numpy ./npz files.

```console
    python3 preprocess.py
```

### Training:
To train the model, you can use the below command. This command will train a model for "vocals" as the target source.
You can also use the "accompaniment" argument instead of "vocals" to train the model for separating "instrumental" track from
the mixed audio signal.
```console
    python3 train.py vocals
```

### Prediction:
For prediction, you can give any mixed audio file and a source target for which the model will make a prediction.
```console
    python3 predict.py accompaniment
```
You can also use "vocals" as source target in the above command.
This will take the first 6s of the input mix audio file and generate two files: ./results/{target}_sample_pred.wav and
./results/mix_sample_downsampled.wav

If you want to separate the entire mixed audio file to a source target you can use the **"--full"** flag in the above command.

```console
    python3 predict.py vocals --full
```

### Evaluation Metrics:
For the mean SDR and SAR metrics for "vocals" and "accompaniment" source targets you can run the below command:

```console
    python3 evaluate.py accompaniment
```