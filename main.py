from waveAnalyzer import *
from linkWavFile import *
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_validate, StratifiedKFold
def linkFIle():
    link_slide_back_file = linkWavFile()
    link_slide_back_file.linkSlideFile('D:\\rfid\\data-devide-main\\data-devide-main\\data\\1.1-1.2backSlidedata\\backSlide')

    link_slide_forward_file = linkWavFile()
    link_slide_forward_file.linkSlideFile('D:\\rfid\\data-devide-main\\data-devide-main\\data\\1.1-1.2forwardSlidedata\\forwardSlide')

    link_dbtouch_file = linkWavFile()
    link_dbtouch_file.linkTouchFile('D:\\rfid\\data-devide-main\\data-devide-main\\data\\1.1-1.2dbtouchdata\\2touch')

    link_touch_file = linkWavFile()
    link_touch_file.linkTouchFile('D:\\rfid\\data-devide-main\\data-devide-main\\data\\1.1-1.2touchdata\\1touch')

    link_pinch_file = linkWavFile()
    link_pinch_file.linkPinchNoneFile('D:\\rfid\\data-devide-main\\data-devide-main\\data\\1.1-1.2pinchdata\\pinch')

    link_none_file = linkWavFile()
    link_none_file.linkPinchNoneFile('D:\\rfid\\data-devide-main\\data-devide-main\\data\\1.1-1.2nonedata\\none')
    
def svmTrain():
    analyzer_1touch = waveAnalyzer()
    analyzer_1touch.devideTouchData('.\\27data1try\\1touch\\1.wav')
    analyzer_1touch.rfftDate()
    analyzer_1touch.setFFTData()
    train_1touch = analyzer_1touch.getTrainFFTData()
    print("1touch", train_1touch.shape)


    analyzer_2touch = waveAnalyzer()
    analyzer_2touch.devideTouchData('.\\27data1try\\2touch\\1.wav')
    analyzer_2touch.rfftDate()
    analyzer_2touch.setFFTData()
    train_2touch = analyzer_2touch.getTrainFFTData()
    print("2touch", train_2touch.shape)


    analyzer_slide_back = waveAnalyzer()
    analyzer_slide_back.devideSlideBackData('.\\27data1try\\backSlide\\1.wav')
    analyzer_slide_back.rfftDate()
    analyzer_slide_back.setFFTData()
    train_slide_back = analyzer_slide_back.getTrainFFTData()
    print("train_slide_back", train_slide_back.shape)

    analyzer_slide_forward = waveAnalyzer()
    analyzer_slide_forward.devideSlideForwardData('.\\27data1try\\forwardSlide\\1.wav')
    analyzer_slide_forward.rfftDate()
    analyzer_slide_forward.setFFTData()
    train_slide_forward = analyzer_slide_forward.getTrainFFTData()
    print("train_slide_forward", train_slide_forward.shape)

    analyzer_pinch = waveAnalyzer()
    analyzer_pinch.devidePinchNOneData('.\\27data1try\\pinch\\1.wav')
    analyzer_pinch.rfftDate()
    analyzer_pinch.setFFTData()
    train_pinch = analyzer_pinch.getTrainFFTData()
    print("train_pinch", train_pinch.shape)

    analyzer_none = waveAnalyzer()
    analyzer_none.devidePinchNoneData('.\\27data1try\\none\\1.wav')
    analyzer_none.rfftDate()
    analyzer_none.setFFTData()
    train_none = analyzer_none.getTrainFFTData()
    print("train_none", train_none.shape)

    clf = svm.SVC(probability=True)
    train_data = np.row_stack((train_1touch[:50], train_2touch[:50], train_slide_back[:50], train_slide_forward[:50], train_pinch[:50], train_none[:50]))
    print(train_data.shape)
    train_lables = [1]*50 + [2]*50 + [3]*50 + [4]*50 + [5]*50 + [6]*50
    clf.fit(train_data, train_lables)
    '''
    scores = cross_validate(clf, train_data, train_lables, cv=5)
    for key, value in scores.items():
        print("{}:{:.2f}+/-{:.2f}".format(key, value.mean(), value.std()))
    '''


    test_data = np.row_stack((train_1touch[50:59], train_2touch[50:59], train_slide_back[50:59], train_slide_forward[50:59], train_pinch[50:59], train_none[50:59]))
    test_lables = [1]*9 + [2]*9 + [3]*9 + [4]*9 + [5]*9 + [6]*9
    predict_lables = clf.predict(test_data)
    sns.set()
    f,ax = plt.subplots()
    cm = confusion_matrix(test_lables, predict_lables, labels=[1, 2, 3, 4, 5, 6])
    sns.heatmap(cm, annot=True,cmap='Blues')

    ax.set_title('confusion matrix')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.show()

    '''
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
    linkFIle()


