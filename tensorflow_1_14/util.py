import librosa
from librosa.util import find_files
from librosa import load

import os, re 
import numpy as np
from config import *

def LoadAudio(file_path) :
    y, sr = load(file_path,sr=SR)
    stft = librosa.stft(y,n_fft=window_size,hop_length=hop_length)
    mag, phase = librosa.magphase(stft)
    mag = mag.astype(np.float32)
    # mag = librosa.amplitude_to_db(mag)
    return mag, phase

# Save Audiofile
def SaveAudio(file_path, mag, phase) :
    # mag = librosa.db_to_amplitude(mag)
    y = librosa.istft(mag*phase,win_length=window_size,hop_length=hop_length)
    librosa.output.write_wav(file_path,y,SR,norm=True)
    print("Save complete!! ", file_path)
    
def SaveSpectrogram(y_mix, y_bass, y_drums, y_other, y_vocal, filename, orig_sr=44100) :
    y_mix = librosa.core.resample(y_mix,orig_sr,SR)
    y_vocal = librosa.core.resample(y_vocal,orig_sr,SR)
    y_bass = librosa.core.resample(y_bass,orig_sr,SR)
    y_drums = librosa.core.resample(y_drums,orig_sr,SR)
    y_other = librosa.core.resample(y_other,orig_sr,SR)

    S_mix = np.abs(librosa.stft(y_mix,n_fft=window_size,hop_length=hop_length)).astype(np.float32)
    S_bass = np.abs(librosa.stft(y_bass,n_fft=window_size,hop_length=hop_length)).astype(np.float32)
    S_drums = np.abs(librosa.stft(y_drums,n_fft=window_size,hop_length=hop_length)).astype(np.float32)
    S_other = np.abs(librosa.stft(y_other,n_fft=window_size,hop_length=hop_length)).astype(np.float32)
    S_vocal = np.abs(librosa.stft(y_vocal,n_fft=window_size,hop_length=hop_length)).astype(np.float32)
    
    norm = S_mix.max()
    S_mix /= norm
    S_drums /= norm
    S_bass /= norm
    S_other /= norm
    S_vocal /= norm
    
    np.savez(os.path.join('./Spectrogram',filename+'.npz'),mix=S_mix, drums=S_drums, bass=S_bass, other=S_other, vocal=S_vocal)
    
def LoadSpectrogram() :
    filelist = find_files('./Spectrogram', ext="npz")
    x_list = []
    y_list = []
    for file in filelist :
        y_element = []
        data = np.load(file)
        x_list.append(data['mix'])
        y_element.append(data['vocal'])
        y_element.append(data['bass'])
        y_element.append(data['drums'])
        y_element.append(data['other'])
        y_list.append(y_element)
    return x_list, y_list


def Magnitude_phase_x(spectrogram) :
    Magnitude_list = []
    Phase_list = []
    for X in spectrogram :
        mag, phase = librosa.magphase(X)
        # mag = librosa.amplitude_to_db(mag)
        Magnitude_list.append(mag)
        Phase_list.append(phase)
    return Magnitude_list, Phase_list

def Magnitude_phase_y(spectrogram) :
    Magnitude_list = []
    Phase_list = []
    for Y in spectrogram :
      mag = []
      phase = []
      for Y_element in Y:
          mag_element, phase_element = librosa.magphase(Y_element)
          # mag_element = librosa.amplitude_to_db(mag_element)
          mag.append(mag_element)
          phase.append(phase_element)
      Magnitude_list.append(mag)
      Phase_list.append(phase)
    return Magnitude_list, Phase_list



def sampling(X_mag,Y_mag) :
    X = []
    y = []
    for mix, target in zip(X_mag,Y_mag) :
        target = np.array(target)
        #starts = np.random.randint(0, mix.shape[1] - patch_size, (mix.shape[1] - patch_size) // SAMPLING_STRIDE)
        starts = np.random.randint(0, target.shape[2] - patch_size, (target.shape[2] - patch_size) // SAMPLING_STRIDE)
        for start in starts:
            end = start + patch_size
            X.append(mix[1:, start:end, np.newaxis])
            y.append(target[:, 1:, start:end])

    X = np.array(X)
    print("X.shape : ", X.shape)
    y = np.array(y)
    y = np.einsum('ijkl->iklj', y)
    print("y.shape : ", y.shape)
    return X, y
