#!/usr/bin/env python
# coding: utf-8

from pydub import AudioSegment
import os
from input_parameter import get_args

args = get_args()
sound_dir = os.path.join(args.DATADIR, 'DSD100/Sources/Dev')
sound_subset_dir = os.path.join(args.DATADIR, 'DSD100subset/Sources/Dev')

def make_ints(sound_dir):
    assert os.path.isdir(sound_dir)
    dir_list = os.listdir(sound_dir)

    for idx_dir in dir_list:
        print(idx_dir)
        assert os.path.isfile(os.path.join(sound_dir, idx_dir, "other.wav"))
        assert os.path.isfile(os.path.join(sound_dir, idx_dir, "drums.wav"))
        assert os.path.isfile(os.path.join(sound_dir, idx_dir, "bass.wav"))

        sound1 = AudioSegment.from_wav(os.path.join(sound_dir, idx_dir, "other.wav"))
        sound2 = AudioSegment.from_wav(os.path.join(sound_dir, idx_dir, "drums.wav"))
        sound3 = AudioSegment.from_wav(os.path.join(sound_dir, idx_dir, "bass.wav"))

        # combined_sounds = sound1 + sound2
        combined_sounds = sound1.overlay(sound2)
        combined_sounds = combined_sounds.overlay(sound3)
        combined_sounds.export(os.path.join(sound_dir, idx_dir, "ints.wav"), format="wav")

make_ints(sound_dir)
make_ints(sound_subset_dir)