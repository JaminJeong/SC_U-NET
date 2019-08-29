# Singing-Voice-Separation
This is an implementation of U-Net for vocal separation with tensorflow

## Requirement
- librosa==0.6.2
- numpy==1.14.3
- tensorflow==1.13.0
- python==3.6.5

## Download Dataset
I download [dsd100](https://sigsep.github.io/datasets/dsd100.html) dataset.
<pre><code>$ python download_data.py --DATADIR ./data </code></pre>

## Make Instrument.wav
I overlap wav files(drum, bass, other) for making instrument wav file.
<pre><code>$ python make_ints.py --DATADIR ./data </code></pre>

## Data
I prepare CCMixter datasets in "./data" and Each track consisted of Mixed, instrumental, Vocal version
<pre><code>$ python CCMixter_process.py --DATADIR ./data </code></pre>

## Usage
- Train
<pre><code>$ python Training.py</code></pre>
- Test
<pre><code>$ python Test.py</code></pre>

## Paper
spectrogram-channels u-net: a source separation model viewing each channel as the spectrogram of each source

## Base Implimentation
* https://github.com/Jeongseungwoo/Singing-Voice-Separation

## To Do List
* convert wav files to mp3 files
* make tfrecord format files
