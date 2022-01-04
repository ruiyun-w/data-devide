import os
import soundfile as sf
import numpy as np
class linkWavFile:
    def linkTouchFile(self, path):
        files = os.listdir(path)
        soundBuffer = []
        for file in files:
            if os.path.splitext(file)[1] == '.wav':
                data, samplerate = sf.read(path + '\\' + file, dtype='float32')
                data = data[192000:]
                data = data[:-192000]
                soundBuffer.extend(data)
        sf.write(path+'\\linkedFile.wav', soundBuffer, samplerate)

    def linkSlideFile(self, path):
        files = os.listdir(path)
        soundBuffer = []
        for file in files:
            if os.path.splitext(file)[1] == '.wav':
                data, samplerate = sf.read(path + '\\' + file, dtype='float32')
                data = data[384000:]
                data = data[:-384000]
                soundBuffer.extend(data)
        sf.write(path+'\\linkedFile.wav', soundBuffer, samplerate)

    def linkPinchNoneFile(self, path):
        files = os.listdir(path)
        soundBuffer = []
        for file in files:
            if os.path.splitext(file)[1] == '.wav':
                data, samplerate = sf.read(path + '\\' + file, dtype='float32')
                data = data[96000:]
                data = data[:-96000]
                soundBuffer.extend(data)
        sf.write(path+'\\linkedFile.wav', soundBuffer, samplerate)





