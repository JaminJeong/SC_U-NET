#!/usr/bin/env python
# coding: utf-8

# In[10]:


import librosa
from librosa.util import find_files
from librosa import load
import os
import numpy as np

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import IPython.display as ipd


# In[11]:


vocal_inst_audio_dir = "/data2/HyundaiPlant/DSD100/Sources/Dev/078 - Moosmusic - Big Dummy Shake"
mix_audio_dir =  "/data2/HyundaiPlant/DSD100/Mixtures/Dev/078 - Moosmusic - Big Dummy Shake"
inst,_ = load(os.path.join(vocal_inst_audio_dir, 'ints.wav'), sr=None)
vocal,_ = load(os.path.join(vocal_inst_audio_dir, 'vocals.wav'), sr=None)
mix,_ = load(os.path.join(mix_audio_dir, 'mixture.wav'), sr=None)


# In[39]:



ipd.Audio(os.path.join(vocal_inst_audio_dir, 'vocals.wav')) # load a local WAV file


# In[12]:


orig_sr=44100
SR =  16000
window_size = 1024
hop_length = 768

patch_size = 128 # roughly 33 seconds

y_mix = librosa.core.resample(mix,orig_sr,SR)
y_vocal = librosa.core.resample(vocal,orig_sr,SR)
y_inst = librosa.core.resample(inst,orig_sr,SR)

S_mix = np.abs(librosa.stft(y_mix,n_fft=window_size,hop_length=hop_length)).astype(np.float32)
S_inst = np.abs(librosa.stft(y_inst,n_fft=window_size,hop_length=hop_length)).astype(np.float32)
S_vocal = np.abs(librosa.stft(y_vocal,n_fft=window_size,hop_length=hop_length)).astype(np.float32)


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import librosa.display


# In[18]:


S_mix.shape


# In[19]:


S_mix[:, 0:256].shape


# In[29]:


# Xdb = S_mix[:, 512:768]
# Xdb = librosa.amplitude_to_db(S_mix[:, 512:768])
Xdb = librosa.amplitude_to_db(S_mix[:, 512:768])
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=SR, x_axis='time', y_axis='hz')


# In[35]:


mag, phase = librosa.magphase(S_mix) 
print("S_mix.shape : ", S_mix.shape)
print("mag.shape : ", mag.shape)
# Xdb = librosa.amplitude_to_db(S_mix)

# df_cm = pd.DataFrame(mag)
# sn.heatmap(df_cm, cmap='coolwarm')

# Xdb = S_mix[:, 512:768]
Xdb = librosa.amplitude_to_db(S_mix[:, 512:768])
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=SR, x_axis='time', y_axis='hz')


# In[6]:


print(vocal)
print(vocal.shape)
print(np.max(vocal))
print(np.min(vocal))

print(y_vocal)
print(y_vocal.shape)
print(np.max(y_vocal))
print(np.min(y_vocal))

print(S_vocal)
print(S_vocal.shape)
print(np.max(S_vocal))
print(np.min(S_vocal))


# In[32]:


plt.figure(figsize=(14, 5))
librosa.display.waveplot(y_vocal, sr=SR)


# In[34]:


import IPython.display as ipd
ipd.Audio(y_vocal, rate=SR)


# In[35]:


plt.figure(figsize=(14, 5))
librosa.display.waveplot(vocal, sr=SR)


# In[36]:


import IPython.display as ipd
ipd.Audio(vocal, rate=SR)


# In[22]:


df_cm = pd.DataFrame(S_inst)
sn.heatmap(df_cm, cmap='coolwarm')


# In[23]:


df_cm = pd.DataFrame(S_vocal)
sn.heatmap(df_cm, cmap='coolwarm')


# In[ ]:

