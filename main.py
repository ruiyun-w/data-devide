import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

#blocksize 24000, 48000Hz, 0.5s
__soundBuffer = []

def devideData():
    data, samplerate = sf.read('D:\\glass\\stethos-py-master\\stethos-py-master\\out\\touch\\1639988883.wav',
                               dtype='float32')
    i = 0
    newBuffer = []
    soundBuffer = []
    touchNumber = 0
    while (i < len(data)):
        if data[i] > 0.25:
            touchNumber = touchNumber + 1
            i = i + 48000
            newBuffer = data[i: i + 47999: 2]
            soundBuffer.append(newBuffer)
        else:
            i = i + 1

if __name__ == '__main__':
    devideData()


'''

x = np.linspace(1,len(newBuffer),len(newBuffer))

plt.plot(x, newBuffer)
plt.show()
'''