import tensorflow as tf
import librosa
from librosa.util import find_files
from librosa import load

from util import *
from config import *
from U_net_deconv_upsampling import U_Net


def main() :
    music_path = "./sample.wav"
    
    mix_wav_mag, mix_wav_phase = LoadAudio(music_path)

    START = 60
    END = START + patch_size  # 11 seconds

    mix_wav_mag=mix_wav_mag[:, START:END]
    mix_wav_phase=mix_wav_phase[:, START:END]

    X = mix_wav_mag[1:].reshape(1,image_width,128,1)
    
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x = {"mag":X},y = None,num_epochs = 1,shuffle = False)

    
    deep_u_net = tf.estimator.Estimator(model_fn=U_Net,model_dir="./model_deconv_upsampling")
    predictions = list(deep_u_net.predict(input_fn=predict_input_fn))
    mask = predictions[0]['outputs']
    mask = np.einsum('ijk->kij', mask)
    
    #target_pred_mag = np.vstack((np.zeros((128)), mask.reshape(512, 128)))
    #SaveAudio(music_path[:-4]+"_vocal.wav",target_pred_mag,mix_wav_phase)
    # vocal
    SaveAudio(music_path[:-4]+"_vocal.wav", np.vstack((np.zeros((128)), mask[0].reshape(image_width, 128))), mix_wav_phase)
    SaveAudio(music_path[:-4]+"_bass.wav", np.vstack((np.zeros((128)), mask[1].reshape(image_width, 128))), mix_wav_phase)
    SaveAudio(music_path[:-4]+"_drums.wav", np.vstack((np.zeros((128)), mask[2].reshape(image_width, 128))), mix_wav_phase)
    SaveAudio(music_path[:-4]+"_other.wav", np.vstack((np.zeros((128)), mask[3].reshape(image_width, 128))), mix_wav_phase)

if __name__ == "__main__" : 
    main()
