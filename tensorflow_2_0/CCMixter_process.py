import librosa
from librosa.util import find_files
from librosa import load
import os
from util import SaveSpectrogram
from input_parameter import get_args

args = get_args()
DSD100_dir = os.path.join(args.DATADIR, 'DSD100')
assert os.path.isdir(DSD100_dir)

mix_dir = os.path.join(DSD100_dir, "Mixtures/Dev")
vocal_inst_dir = os.path.join(DSD100_dir, "Sources/Dev")

# Save Spectrogram 
def CCMixter() : 
    '''
    mix : original wav file
    inst : inst wav file
    vocal : vocal wac file
    '''
    Audiolist = os.listdir(vocal_inst_dir)
    print("Audiolist : ", Audiolist)
    for audio in Audiolist :
        try :
            mix_audio_dir = os.path.join(mix_dir, audio)
            vocal_inst_audio_dir = os.path.join(vocal_inst_dir, audio)
            assert os.path.isdir(mix_audio_dir)
            assert os.path.isdir(vocal_inst_audio_dir)
            print("Song : %s" % audio)
            if os.path.exists(os.path.join('./Spectrogram',audio+'.npz')) :
                print("Already exist!! Skip....")
                continue

            bass,_ = load(os.path.join(vocal_inst_audio_dir, 'bass.wav'), sr=None)
            drums,_ = load(os.path.join(vocal_inst_audio_dir, 'drums.wav'), sr=None)
            other,_ = load(os.path.join(vocal_inst_audio_dir, 'other.wav'), sr=None)
            vocal,_ = load(os.path.join(vocal_inst_audio_dir, 'vocals.wav'), sr=None)
            mix,_ = load(os.path.join(mix_audio_dir, 'mixture.wav'), sr=None)
            print("Saving...")
    
            SaveSpectrogram(mix, bass, drums, other, vocal, audio)
        except IndexError as e :
            print("Wrong Directory")
            pass

if __name__ == '__main__' :
    CCMixter()
    print("Complete!!!!")
