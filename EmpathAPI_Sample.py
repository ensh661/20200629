# -*- coding: utf-8 -*-
from poster.encode import multipart_encode, MultipartParam
from poster.streaminghttp import register_openers
import urllib2
import pydub
import wave
import numpy as np
import scipy
import scipy.signal
from scipy import fromstring, int16
import struct
import math
import os

def readWav(filename):
    """
    wavファイルを読み込んで，データ・サンプリングレートを返す関数
    """
    # 読み込み
    wf = wave.open(filename)
    # サンプリング周波数
    fs = wf.getframerate()
    # チャネル数
    ch = wf.getnchannels()
    # バイト数．2なら2bytes(16bit)
    width = wf.getsampwidth()
    # データ点の数
    fn = wf.getnframes()
    params = wf.getparams()

    # -1 ~ 1までに正規化した信号データを読み込む
    data = np.frombuffer(wf.readframes(wf.getnframes()), dtype="int16") / 32768.0
    return (data, fs, ch, width, fn, params)


def upsampling(conversion_rate, data, fs):
    """
    アップサンプリングを行う．
    入力として，変換レートとデータとサンプリング周波数．
    アップサンプリング後のデータとサンプリング周波数を返す．
    """
    # 補間するサンプル数を決める
    interpolationSampleNum = conversion_rate - 1

    # FIRフィルタの用意をする
    nyqF = fs / 2.0  # 変換後のナイキスト周波数
    cF = (fs / 2.0 - 500.) / nyqF  # カットオフ周波数を設定（変換前のナイキスト周波数より少し下を設定）
    taps = 511  # フィルタ係数（奇数じゃないとだめ）
    b = scipy.signal.firwin(taps, cF)  # LPFを用意

    # 補間処理
    upData = []
    for d in data:
        upData.append(d)
        # 1サンプルの後に，interpolationSampleNum分だけ0を追加する
        for i in range(interpolationSampleNum):
            upData.append(0.0)

    # フィルタリング
    resultData = scipy.signal.lfilter(b, 1, upData)
    return (resultData, fs * conversion_rate)


def downsampling(conversion_rate, data, fs):
    """
    ダウンサンプリングを行う．
    入力として，変換レートとデータとサンプリング周波数．
    ダウンサンプリング後のデータとサンプリング周波数を返す．
    """
    # 間引くサンプル数を決める
    decimationSampleNum = conversion_rate - 1

    # FIRフィルタの用意をする
    nyqF = fs / 2.0  # 変換後のナイキスト周波数
    cF = (fs / conversion_rate / 2.0 - 500.) / nyqF  # カットオフ周波数を設定（変換前のナイキスト周波数より少し下を設定）
    taps = 511  # フィルタ係数（奇数じゃないとだめ）
    b = scipy.signal.firwin(taps, cF)  # LPFを用意

    # フィルタリング
    data = scipy.signal.lfilter(b, 1, data)

    # 間引き処理
    downData = []
    for i in range(0, len(data), decimationSampleNum + 1):
        downData.append(data[i])

    return (downData, fs / conversion_rate)


def writeWav(filename, data, fs):
    """
    入力されたファイル名でwavファイルを書き出す．
    """
    # データを-32768から32767の整数値に変換
    data = [int(x * 32767.0) for x in data]
    # バイナリ化
    binwave = struct.pack("h" * len(data), *data)
    wf = wave.Wave_write(filename)
    wf.setparams((
        1,  # channel
        2,  # byte width
        fs,  # sampling rate
        len(data),  # number of frames
        "NONE", "not compressed"  # no compression
    ))
    wf.writeframes(binwave)
    wf.close()

def cut_wav(filename,time):
    # timeの単位は[sec]
    # ファイルを読み出し
    #wavf = filename + '.wav'
    wavf = filename
    wr = wave.open(wavf, 'r')

    # waveファイルが持つ性質を取得
    ch = wr.getnchannels()
    width = wr.getsampwidth()
    fr = wr.getframerate()
    fn = wr.getnframes()
    total_time = 1.0 * fn / fr
    integer = math.floor(total_time) # 小数点以下切り捨て
    t = int(time)  # 秒数[sec]
    frames = int(ch * fr * t)
    num_cut = int(integer//t)

    data = wr.readframes(wr.getnframes())
    wr.close()
    X = fromstring(data, dtype=int16)

    for i in range(num_cut):
        # print(i)
        # 出力データを生成
        outf = '001_bunkatsu/' + str(i) + '.wav'
        start_cut = i*frames
        end_cut = i*frames + frames
        # print(start_cut)
        # print(end_cut)
        Y = X[start_cut:end_cut]
        outd = struct.pack("h" * len(Y), *Y)

        # 書き出し
        ww = wave.open(outf, 'w')
        ww.setnchannels(ch)
        ww.setsampwidth(width)
        ww.setframerate(fr)
        ww.writeframes(outd)
        ww.close()

    return num_cut

if __name__ == "__main__":
    #print("input filename = ")
    filename = "001.mp3"
    #filename = input()
    # 何倍にするかを決めておく
    #up_conversion_rate = 4
    # 何分の1にするか決めておく．ここではその逆数を指定しておく（例：1/2なら2と指定）
    down_conversion_rate = 4

    mp3file = pydub.AudioSegment.from_file("001.mp3")
    wavfile = mp3file.export("001.wav", format="wav")

    # テストwavファイルを読み込む
    data, fs, ch, width, fn, params = readWav("001.wav")
    print "Channel: ", ch
    print "Sample width: ", width
    print "Frame Rate: ", fs
    print "Frame num: ", fn
    print "Params: ", params
    print "Total time: ", 1.0 * fn / fs
    X = fromstring(data, dtype=int16)
    # print(X)
    #upData, upFs = upsampling(up_conversion_rate, data, fs)
    # ダウンサンプリング
    downData, downFs = downsampling(down_conversion_rate, data, fs)

    #writeWav("up.wav", upData, upFs)
    writeWav("001_down.wav", downData, downFs)

    # 分割
    # 一応既に同じ名前のディレクトリがないか確認。
    file = os.path.exists("001_bunkatsu")
    # print(file)
    if file == False:
        # 保存先のディレクトリの作成
        os.mkdir("001_bunkatsu")
    # 5秒で分割
    num_cut = cut_wav("001_down.wav", 5)

    for i in range(num_cut):
        print "n=", i
        url = "https://api.webempath.net/v2/analyzeWav"
        register_openers()
        items = []
        items.append(MultipartParam('apikey', "CNqfAmuTAA914F_p9ayy50PoBnMm3eMVRuULz6HRoG8"))
        items.append(MultipartParam.from_file('wav', '001_bunkatsu/' + str(i) + '.wav'))
        datagen, headers = multipart_encode(items)
        request = urllib2.Request(url, datagen, headers)
        response = urllib2.urlopen(request)
        if response.getcode() == 200:
            print(response.read())
        else:
            print("HTTP status %d" % (response.getcode()))
        i = i + 1
