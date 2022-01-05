from waveAnalyzer import *
from linkWavFile import *
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_validate, StratifiedKFold
def linkFIle():

    link_slide_back_file = linkWavFile()
    link_slide_back_file.linkSlideFile('.\\data\\1.1-1.2backSlidedata\\backSlide')


    link_slide_forward_file = linkWavFile()
    link_slide_forward_file.linkSlideFile('.\\data\\1.1-1.2forwardSlidedata\\forwardSlide')

    link_dbtouch_file = linkWavFile()
    link_dbtouch_file.linkTouchFile('.\\data\\1.1-1.2dbtouchdata\\2touch')

    link_touch_file = linkWavFile()
    link_touch_file.linkTouchFile('.\\data\\1.1-1.2touchdata\\1touch')

    link_pinch_file = linkWavFile()
    link_pinch_file.linkPinchNoneFile('.\\data\\1.1-1.2pinchdata\\pinch')

    link_none_file = linkWavFile()
    link_none_file.linkPinchNoneFile('.\\data\\1.1-1.2nonedata\\none')

def dataPre():
    # analyzer_1touch = waveAnalyzer()
    # analyzer_1touch.getTouchPeaks('.\\data\\1.1-1.2touchdata\\1touch\\touch_linkedFile.wav')
    # analyzer_1touch.rfftDate()
    # analyzer_1touch.setFFTData()
    # analyzer_1touch.outPutJSON('touch.json')
    #
    #
    # analyzer_2touch = waveAnalyzer()
    # analyzer_2touch.getTouchPeaks('.\\data\\1.1-1.2dbtouchdata\\2touch\\dbtouch_linkedFile.wav')
    # analyzer_2touch.rfftDate()
    # analyzer_2touch.setFFTData()
    # analyzer_2touch.outPutJSON('dbtouch.json')
    #
    #
    # analyzer_slide_back = waveAnalyzer()
    # analyzer_slide_back.getSlidePeaks('.\\data\\1.1-1.2backSlidedata\\backSlide\\back_linkedFile.wav')
    # analyzer_slide_back.rfftDate()
    # analyzer_slide_back.setFFTData()
    # analyzer_slide_back.outPutJSON('back.json')
    #
    # analyzer_slide_forward = waveAnalyzer()
    # analyzer_slide_forward.getSlidePeaks('.\\data\\1.1-1.2forwardSlidedata\\forwardSlide\\forward_linkedFile.wav')
    # analyzer_slide_forward.rfftDate()
    # analyzer_slide_forward.setFFTData()
    # analyzer_slide_forward.outPutJSON('forward.json')

    analyzer_pinch = waveAnalyzer()
    analyzer_pinch.devidePinchNoneData('.\\data\\1.1-1.2pinchdata\\pinch\\pinch_linkedFile.wav')
    analyzer_pinch.rfftDate()
    analyzer_pinch.setFFTData()
    analyzer_pinch.outPutJSON('pinch.json')

    analyzer_none = waveAnalyzer()
    analyzer_none.devidePinchNoneData('.\\data\\1.1-1.2nonedata\\none\\none_linkedFile.wav')
    analyzer_none.rfftDate()
    analyzer_none.setFFTData()
    analyzer_none.outPutJSON('none.json')




def svmTrain():
    clf = svm.SVC(probability=True)
    touchFile = open('touch.json')
    touchData = json.load(touchFile)

    dbtouchFile = open('dbtouch.json')
    dbtouchData = json.load(dbtouchFile)

    backSlideFile = open('back.json')
    backSlideData = json.load(backSlideFile)

    forwardSlideFile = open('forward.json')
    forwardSlideData = json.load(forwardSlideFile)

    pinchFile = open('pinch.json')
    pinchData = json.load(pinchFile)

    noneFile = open('none.json')
    noneData = json.load(noneFile)

    train_data = np.row_stack((touchData[:719], dbtouchData[:719], backSlideData[:719], forwardSlideData[:719], pinchData[:719], noneData[:719]))
    train_lables = ['touch']*719 + ['dbtouch']*719 + ['backSlide']*719 + ['forwardSlide']*719 + ['pinch']*719 + ['noneData']*719
    # train_dic ={'data':train_data, 'labels':train_lables }
    # with open('train_data.json', 'w') as f:
    #     json.dump(str(train_dic), f)

    clf.fit(train_data, train_lables)
    filename = 'trained_model.sav'
    pickle.dump(clf, open(filename, 'wb'))


def svmTest():
    touchFile = open('touch.json')
    touchData = json.load(touchFile)

    dbtouchFile = open('dbtouch.json')
    dbtouchData = json.load(dbtouchFile)

    backSlideFile = open('back.json')
    backSlideData = json.load(backSlideFile)

    forwardSlideFile = open('forward.json')
    forwardSlideData = json.load(forwardSlideFile)

    pinchFile = open('pinch.json')
    pinchData = json.load(pinchFile)

    noneFile = open('none.json')
    noneData = json.load(noneFile)

    test_data = np.row_stack((touchData[719:819], dbtouchData[719:819], backSlideData[719:819], forwardSlideData[719:819],
                               pinchData[719:819], noneData[719:819]))
    test_lables = ['touch'] * 100 + ['dbtouch'] * 100 + ['backSlide'] * 100 + ['forwardSlide'] * 100 + [
        'pinch'] * 100 + ['noneData'] * 100

    clf = pickle.load(open('trained_model.sav', 'rb'))
    predict_lables = clf.predict(test_data)

    sns.set()
    f, ax = plt.subplots()
    cm = confusion_matrix(test_lables, predict_lables)
    sns.heatmap(cm, annot=True, cmap='Blues')
    ax.set_title('confusion matrix')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.show()


    '''
    scores = cross_validate(clf, train_data, train_lables, cv=5)
    for key, value in scores.items():
        print("{}:{:.2f}+/-{:.2f}".format(key, value.mean(), value.std()))


    test_1touch_label = clf.predict(train_1touch[:50])
    test_1touch_proba = clf.predict_proba(train_1touch[:50])
    test_1touch_y = [1]*50
    test_1touch_score = accuracy_score(test_1touch_label.reshape(-1,1), test_1touch_y)
    print(test_1touch_label, test_1touch_score, test_1touch_proba)

    test_2touch_label = clf.predict(train_2touch[:50])
    test_2touch_proba = clf.predict_proba(train_2touch[:50])
    test_2touch_y = [2]*50
    test_2touch_score = accuracy_score(test_2touch_label.reshape(-1,1), test_2touch_y)
    print(test_2touch_label, test_2touch_score, test_2touch_proba)

    test_slide_back_label = clf.predict(train_slide_back[:50])
    test_slide_back_proba = clf.predict_proba(train_slide_back[:50])
    test_slide_back_y = [3]*50
    test_slide_back_score = accuracy_score(test_slide_back_label.reshape(-1,1), test_slide_back_y)
    print(test_slide_back_label, test_slide_back_score, test_slide_back_proba)

    test_slide_forward_label = clf.predict(train_slide_forward[:50])
    test_slide_forward_proba = clf.predict_proba(train_slide_forward[:50])
    test_slide_forward_y = [4]*50
    test_slide_forward_score = accuracy_score(test_slide_forward_label.reshape(-1,1), test_slide_forward_y)
    print(test_slide_forward_label, test_slide_forward_score, test_slide_forward_proba)

    test_pinch_label = clf.predict(train_pinch[:50])
    test_pinch_proba = clf.predict_proba(train_pinch[:50])
    test_pinch_y = [5]*50
    test_pinch_score = accuracy_score(test_pinch_label.reshape(-1,1), test_pinch_y)
    print(test_pinch_label, test_pinch_score, test_pinch_proba)

    test_none_label = clf.predict(train_none[:50])
    test_none_proba = clf.predict_proba(train_none[:50])
    test_none_y = [6]*50
    test_none_score = accuracy_score(test_none_label.reshape(-1,1), test_none_y)
    print(test_none_label, test_none_score, test_none_proba)
    '''

if __name__ == '__main__':
    svmTest()



