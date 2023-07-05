import pyaudio
import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import write

import warnings
warnings.simplefilter('ignore')

import whisper


def whisper_model(audio_data):
    model = whisper.load_model("small")
    # audio_file = "/home/gaku/whisper/001.mp3"
    result = model.transcribe(audio_data)
    translate_result = model.transcribe(audio_data,task="translate")

    return result, translate_result


def record(idx, sr, framesize, t):
    print("start")
    pa = pyaudio.PyAudio() 
    data = [] 
    dt = 1 / sr 

    # start stream
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sr,
                     input=True, input_device_index=idx, frames_per_buffer=framesize)

    # recode
    for i in range(int(((t / dt) / framesize))): 
        frame = stream.read(framesize) 
        data.append(frame)

    # finish stream
    stream.stop_stream() 
    stream.close()
    pa.terminate()
    
    # summary fream 
    data = b"".join(data)

    data = np.frombuffer(data, dtype="int16")
    data_show = np.frombuffer(data, dtype="int16") / float((np.power(2, 16) / 2) - 1)

    return data, data_show, i



if __name__ == "__main__":
    sr = 44100        # サンプリングレート
    framesize = 1024  # フレームサイズ
    idx = 1          # マイクのチャンネル
    t = 10             # 計測時間[s]

    data, data_show, i = record(idx, sr, framesize, t)
    print("finish")
    write("test.wav", sr, data)
    data = "./test.wav"
    result, translate_result = whisper_model(data)

    if result["language"] == "ja":
        lan = "Japanese"
        print(lan)
    else:
        print(result["language"])

    print(result["text"])
    print("")
    print("English")
    print(translate_result["text"])






