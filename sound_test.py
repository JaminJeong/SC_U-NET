import librosa
from librosa.util import find_files
from librosa import load
import numpy as np

from config import *

def LoadAudio(file_path) :
    y, sr = load(file_path,sr=SR)
    stft = librosa.stft(y,n_fft=window_size,hop_length=hop_length)
    print("stft : ", stft)
    print("stft[0][0] : ", type(stft[0][0]))
    # print("stft.type : ", stft.dtype())
    # print("stft.shape : ", stft.shape)

    # mag, phase = librosa.magphase(stft)
    # mag = mag.astype(np.float32)
    # mag = librosa.amplitude_to_db(mag)

    # return mag, phase
    return stft

# Save raw Audiofile
def SaveRawAudio(file_path, mag) :
    mag = librosa.db_to_amplitude(mag)
    y = librosa.istft(mag,win_length=window_size,hop_length=hop_length)
    librosa.output.write_wav(file_path,y,SR,norm=True)
    print("Save : ", file_path)

# Save Audiofile
def SaveAudio(file_path, mag, phase) :
    mag = librosa.db_to_amplitude(mag)
    y = librosa.istft(mag*phase,win_length=window_size,hop_length=hop_length)
    librosa.output.write_wav(file_path,y,SR,norm=True)
    print("Save complete!!")

START = 60
END = START + patch_size  # 11 seconds

music_path = "./sample.wav"
vocal_path = "./vocals.wav"

mix_wav_stft = LoadAudio(music_path)

# mix_wav_stft = mix_wav_stft[:, START:END]
# SaveRawAudio(music_path[:-4] + "_copy.wav", mix_wav_stft)


# mix_wav_mag, mix_wav_phase = LoadAudio(music_path)
# vocal_wav_mag, vocal_wav_phase = LoadAudio(vocal_path)

# mix_wav_mag = mix_wav_mag[:, START:END]
# mix_wav_phase = mix_wav_phase[:, START:END]
#
# vocal_wav_mag = vocal_wav_mag[:, START:END]
# vocal_wav_phase = vocal_wav_phase[:, START:END]
#
# # SaveRawAudio(music_path[:-4] + "_copy.wav", mix_wav_mag, mix_wav_phase)
# SaveAudio(music_path[:-4] + "_copy.wav", mix_wav_mag, mix_wav_phase)
# SaveAudio(vocal_path[:-4] + "_copy.wav", vocal_wav_mag, mix_wav_phase)
