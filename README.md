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
Andreas Jansson, et al. SINGING VOICE SEPARATION WITH DEEP U-NET CONVOLUTIONAL NETWORKS. 2017. <br> paper: https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf

## Original Implimentation
* https://github.com/Jeongseungwoo/Singing-Voice-Separation
