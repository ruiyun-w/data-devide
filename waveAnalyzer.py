import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_validate, StratifiedKFold

class waveAnalyzer:
    def __init__(self):
        #blocksize 24000, 48000Hz, 0.5s
        self.__soundBuffer = []
        self.__axis = np.fft.rfftfreq(48000, d=1.0 / 48000)
        self.__fftData = None
        self.__trainFFTDate = None
        #选取6kHz到22kHZ的数据，要check
        self.__minFreq = 6000
        self.__maxFreq = 24000

    def devideTouchData(self, file):
        data, samplerate = sf.read(file, dtype='float32')
        i = 0
        newBuffer = []
        soundBuffer = []
        touchNumber = 0
        while (i < len(data)):
            if data[i] > 0.25:
                touchNumber = touchNumber + 1
                newBuffer = data[i: i + 96000: 2]
                soundBuffer.append(newBuffer)
                i = i + 96000
            else:
                i = i + 1
        self.__soundBuffer = soundBuffer
        print(touchNumber)



    def devideSlideForwardData(self, file):
        data, samplerate = sf.read(file, dtype='float32')
        i = 0
        newBuffer = []
        soundBuffer = []
        touchNumber = 0
        while (i < len(data)):
            if data[i] > 0.26:
                touchNumber = touchNumber + 1
                newBuffer = data[i: i + 96000: 2]
                soundBuffer.append(newBuffer)
                i = i + 288000
            else:
                i = i + 1
        self.__soundBuffer = soundBuffer
        print(touchNumber)

    def devideSlideBackData(self, file):
        data, samplerate = sf.read(file, dtype='float32')
        i = 0
        newBuffer = []
        soundBuffer = []
        touchNumber = 0
        while (i < len(data)):
            if data[i] > 0.2:
                touchNumber = touchNumber + 1
                newBuffer = data[i: i + 96000: 2]
                soundBuffer.append(newBuffer)
                i = i + 288000
            else:
                i = i + 1
        self.__soundBuffer = soundBuffer
        print(touchNumber)
        '''
                x = np.linspace(1,len(data),len(data))
                plt.plot(x, data)
                plt.show()
                print(touchNumber)
        '''

    def devidePinchNoneData(self, file):
        data, samplerate = sf.read(file, dtype='float32')
        i = 0
        newBuffer = []
        soundBuffer = []
        while (i < len(data)):
            newBuffer = data[i: i + 96000: 2]
            soundBuffer.append(newBuffer)
            i = i + 96000
        self.__soundBuffer = soundBuffer


    def rfftDate(self):
        soundBuffer = self.getSoundBuffer()
        newAxis = self.getAxis()
        newFFTData = np.empty_like(newAxis)
        #最后一次数据不足0.5s, 故舍去
        for i in range(len(soundBuffer) - 1):
            newFFTData = np.log10(
                np.abs(
                    np.fft.rfft(
                        np.hamming(len(soundBuffer[i])) * soundBuffer[i]
                    )
                )
            )
            newFFTData = np.convolve(newFFTData, np.ones(5) / 5, mode='same')
            if self.__fftData is None:
                self.__fftData = newFFTData
            else:
                self.__fftData = np.row_stack((self.__fftData, newFFTData))

    def setFFTData(self):
        minFreq, maxFreq = self.getSweepRange()
        axis = self.getAxis()
        fftData = self.getFFTData()
        left = np.searchsorted(axis, minFreq, side='left')
        right = np.searchsorted(axis, maxFreq, side='left')
        for i in range(len(fftData)):
            newTrainFFTData = fftData[i][left:right]
            if self.__trainFFTDate is None:
                self.__trainFFTDate = newTrainFFTData
            else:
                self.__trainFFTDate = np.row_stack((self.__trainFFTDate, newTrainFFTData))

    def getFFTData(self):
        return self.__fftData

    def getAxis(self):
        return self.__axis

    def getSoundBuffer(self):
        return self.__soundBuffer

    def getSweepRange(self):
        return self.__minFreq, self.__maxFreq

    def getTrainFFTData(self):
        return self.__trainFFTDate

