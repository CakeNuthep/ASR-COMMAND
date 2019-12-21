import os
import argparse

import numpy as np
from scipy.io import wavfile
from hmmlearn import hmm
from python_speech_features import mfcc
from HMMTrainer import HMMTrainer
from readMicrophone import TapTester
import readMicrophone as mic
from KeyCodeModel import KeyCodeModel as k
import sendKeyCode as sendKey
import time

hmm_models = []



def listenCommand(wave,rate):
    # Extract MFCC features
    mfcc_features = mfcc(wave, rate)
    # Define variables
    max_score = None
    output_label = None

    # Iterate through all HMM models and pick
    # the one with the highest score
    for item in hmm_models:
        hmm_model, label = item
        score = hmm_model.get_score(mfcc_features)
        if max_score is None or score > max_score:
            max_score = score
            output_label = label

   
    print("Predicted:", output_label)
    
    #send keyCode
    keyCode = int(output_label.split('\\')[1])
    if keyCode == k.VK_DOWN:
        sendKey.PressKey(k.VK_DOWN)
    elif keyCode == k.VK_UP:
        sendKey.ReleaseKey(k.VK_DOWN)
        sendKey.PressKey(k.VK_UP)   
        time.sleep(0.5)
        sendKey.ReleaseKey(k.VK_UP) 


if __name__=='__main__':
    #args = build_arg_parser().parse_args()
    #input_folder = args.input_folder
    input_folder = "./Sound"
    # Parse the input directory
    for dirname in os.listdir(input_folder):
        # Get the name of the subfolder
        subfolder = os.path.join(input_folder, dirname)

        if not os.path.isdir(subfolder):
            continue

        # Extract the label
        label = subfolder[subfolder.rfind('/') + 1:]
        # Initialize variables
        X = np.array([])
        y_words = []
        # Iterate through the audio files (leaving 1 file for testing in each class)
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
            # Read the input file
            filepath = os.path.join(subfolder, filename)
            sampling_freq, audio = wavfile.read(filepath)
            # Extract MFCC features
            mfcc_features = mfcc(audio, sampling_freq, nfft=1200)
            # Append to the variable X
            if len(X) == 0:
                X = mfcc_features
            else:
                X = np.append(X, mfcc_features, axis=0)
                # Append the label
                y_words.append(label)
                # Train and save HMM model
                hmm_trainer = HMMTrainer(model_name='GMMHMM',n_mix=3)
                hmm_trainer.train(X)
                hmm_models.append((hmm_trainer, label))
                hmm_trainer = None

    
   
    tt = TapTester()
    while(True):
        tt.listenCommand(listenCommand)