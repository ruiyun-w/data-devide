import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn import svm
class waveAnalyzer:
    def __init__(self):
        #blocksize 24000, 48000Hz, 0.5s
        self.__soundBuffer = []
        self.__axis = np.fft.rfftfreq(24000, d=1.0 / 48000)
        self.__fftData = None
        self.__trainFFTDate = None
        #选取6kHz到22kHZ的数据，要check
        self.__minFreq = 6000
        self.__maxFreq = 22000

    def devideData(self, file):
        data, samplerate = sf.read(file, dtype='float32')
        i = 0
        newBuffer = []
        soundBuffer = []
        touchNumber = 0
        while (i < len(data)):
            if data[i] > 0.25:
                touchNumber = touchNumber + 1
                newBuffer = data[i: i + 47999: 2]
                soundBuffer.append(newBuffer)
                i = i + 48000
            else:
                i = i + 1
        self.__soundBuffer = soundBuffer
        '''
        x = np.linspace(1,len(soundBuffer[0]),len(soundBuffer[0]))
        plt.plot(x, soundBuffer[0])
        plt.show()
        print(touchNumber)
        '''
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

if __name__ == '__main__':
    analyzer1 = waveAnalyzer()
    analyzer1.devideData('1touch.wav')
    analyzer1.rfftDate()
    fftData1 = analyzer1.getFFTData()
    analyzer1.setFFTData()
    trainFFTData1 = analyzer1.getFFTData()
    print(trainFFTData1.shape)

    analyzer2 = waveAnalyzer()
    analyzer2.devideData('2touch.wav')
    analyzer2.rfftDate()
    fftData2 = analyzer2.getFFTData()
    analyzer2.setFFTData()
    trainFFTData2 = analyzer2.getFFTData()
    print(trainFFTData2.shape)

    clf = svm.SVC(probability=True)
    trainFFTData = np.row_stack((trainFFTData1[:13], trainFFTData2[:9]))
    print(trainFFTData.shape)
    lables = [1] * 13 + [2] *9
    clf.fit(trainFFTData, lables)

    test1touchData = trainFFTData1[-1]
    test2touchData = trainFFTData2[-1]
    test1touchLabel = clf.predict(test1touchData.reshape(1,-1))
    test2touchLabel = clf.predict(test2touchData.reshape(1,-1))
    test1touchProba = clf.predict_proba(test1touchData.reshape(1,-1))
    test2touchProba = clf.predict_proba(test2touchData.reshape(1,-1))


    print(test1touchLabel, test1touchProba)
    print(test2touchLabel, test2touchProba)






