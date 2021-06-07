import matplotlib
import pyaudio
import wave
import os
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np
import scipy.io
import scipy.io.wavfile
def saveAudio(filename, data):
    waveFile = wave.open(filename, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(data)
    waveFile.close()
    stream.stop_stream()

def recordAudio(archivo, stream):
    stream.start_stream()
    print("grabando...")
    frames=[]
    for i in range(0, int(RATE/CHUNK*duracion)):
        data=stream.read(CHUNK)
        frames.append(data)
    print("grabación terminada")
    saveAudio(archivo, b''.join(frames))

def plotAudio(archivo1, archivo2):
    sampleRate, audioBuffer1 = scipy.io.wavfile.read(archivo1)
    sampleRate, audioBuffer2 = scipy.io.wavfile.read(archivo2)
    
    duration = len(audioBuffer1)/sampleRate
    time = np.arange(0,duration,1/sampleRate) #time vector
    plt.plot(time,audioBuffer1, label="Audio 1")
    plt.plot(time,audioBuffer2, label="Audio 2")
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title("Audio 1 & Audio 2")
    plt.show()

def wavesSum(archivo1, archivo2, filename):
    sampleRate, data1 = scipy.io.wavfile.read(archivo1)
    sampleRate, data2 = scipy.io.wavfile.read(archivo2)
    wavesSum = np.int16([0 for i in range(len(data1))])
    for i in range(0, len(data1)):
        ### read 1 frame and the position will updated ###
        wavesSum[i] = data1[i]+data2[i]

    duration = len(wavesSum)/sampleRate
    time = np.arange(0,duration,1/sampleRate) #time vector
    plt.plot(time,wavesSum, label="Suma")
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title("Suma de Audio 1 y Audio 2")
    plt.show()
    scipy.io.wavfile.write(filename, sampleRate, wavesSum.astype(np.int16))

FORMAT=pyaudio.paInt16
CHANNELS=1
RATE=44100
CHUNK=1025
duracion=4
dirname = os.path.dirname(__file__)
archivo1= os.path.join(dirname, "audio1.wav")
archivo2= os.path.join(dirname, "audio2.wav")
suma= os.path.join(dirname, "suma.wav")

audio=pyaudio.PyAudio()
stream=audio.open(format=FORMAT,channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

input("Press Enter to record first audio...")
recordAudio(archivo1, stream)
input("Press Enter to record second audio...")
recordAudio(archivo2, stream)

plotAudio(archivo1, archivo2)

wavesSum(archivo1, archivo2, suma)
#DETENEMOS GRABACIÓN
stream.close()
audio.terminate()


