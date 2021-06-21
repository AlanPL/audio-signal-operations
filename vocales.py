import time
from numpy.lib.function_base import average
import pyaudio
import wave
import os
import matplotlib.pyplot as plt
import scipy.io.wavfile
import scipy.io
from scipy.fft import fft, fftfreq, fftshift
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


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
    frames = []
    for i in range(0, int(RATE/CHUNK*duracion)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Grabación finalizada")
    saveAudio(archivo, b''.join(frames))


def playAudio(filename):
    wf = wave.open(filename, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    # read data (based on the chunk size)
    data = wf.readframes(CHUNK)
    # play stream (looping from beginning of file to the end)
    while data:
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()
    p.terminate()


def plotTresAudios(archivo1, archivo2, archivo3):
    sampleRate, audioBuffer1 = scipy.io.wavfile.read(archivo1)
    sampleRate, audioBuffer2 = scipy.io.wavfile.read(archivo2)
    sampleRate, audioBuffer3 = scipy.io.wavfile.read(archivo3)

    print('Longitud del arreglo de datos de cada audio: ', len(audioBuffer1))
    duration = len(audioBuffer1)/sampleRate
    time = np.arange(0, duration, 1/sampleRate)  # time vector
    originalPlot = plt.figure(1)

    plt.subplot(3, 1, 1)
    plt.plot(time, audioBuffer1, '#e36b2c', label="Audio 1")
    plt.ylabel('Amplitud')
    plt.title("Audio 1, Audio 2 & Resultado")

    plt.subplot(3, 1, 2)
    plt.plot(time, audioBuffer2, '#6dc36d', label="Audio 2")
    plt.ylabel('Amplitud')

    plt.subplot(3, 1, 3)
    plt.plot(time, audioBuffer3, '#109dfa', label="Resultado")

    plt.legend()
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.show()


def plotDosAudios(archivo1, archivo2):
    sampleRate, audioBuffer1 = scipy.io.wavfile.read(archivo1)
    sampleRate, audioBuffer2 = scipy.io.wavfile.read(archivo2)

    print('Longitud del arreglo de datos de cada audio: ', len(audioBuffer1))
    duration = len(audioBuffer1)/sampleRate
    time = np.arange(0, duration, 1/sampleRate)  # time vector
    originalPlot = plt.figure(1)

    plt.subplot(2, 1, 1)
    plt.plot(time, audioBuffer1, '#6dc36d', label="Audio 1")
    plt.ylabel('Amplitud')
    plt.title("Audio 1 & Resultado")

    plt.subplot(2, 1, 2)
    plt.plot(time, audioBuffer2, '#109dfa', label="Audio 2")

    plt.legend()
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.show()


def plotAudios(archivo1, archivo2):
    sampleRate, audioBuffer1 = scipy.io.wavfile.read(archivo1)
    sampleRate, audioBuffer2 = scipy.io.wavfile.read(archivo2)
    print('Longitud del arreglo de datos de cada audio: ', len(audioBuffer1))
    duration = len(audioBuffer1)/sampleRate
    time = np.arange(0, duration, 1/sampleRate)  # time vector
    originalPlot = plt.figure(1)
    plt.plot(time, audioBuffer1, label="Audio 1")
    plt.plot(time, audioBuffer2, label="Audio 2")
    plt.legend()
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.title("Audio 1 & Audio 2")
    plt.show()


def plotAudio(archivo):
    global figureIndex
    figureIndex += 1
    sampleRate, audioBuffer = scipy.io.wavfile.read(archivo)
    print('Longitud del arreglo de datos: ', len(audioBuffer))
    duration = len(audioBuffer)/sampleRate
    time = np.arange(0, duration, 1/sampleRate)  # time vector
    figure = plt.figure(figureIndex)
    plt.plot(time, audioBuffer, label="Audio")
    plt.legend()
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.title("Audio")
    plt.title(os.path.basename(archivo))
    plt.show()


def maxFrequency(X, F_sample, Low_cutoff=80, High_cutoff=300):
    """ Searching presence of frequencies on a real signal using FFT
    Inputs
    =======
    X: 1-D numpy array, the real time domain audio signal (single channel time series)
    Low_cutoff: float, frequency components below this frequency will not pass the filter (physical frequency in unit of Hz)
    High_cutoff: float, frequency components above this frequency will not pass the filter (physical frequency in unit of Hz)
    F_sample: float, the sampling frequency of the signal (physical frequency in unit of Hz)
    """

    M = X.size  # let M be the length of the time series
    Spectrum = fft(X, n=M)
    [Low_cutoff, High_cutoff, F_sample] = map(
        float, [Low_cutoff, High_cutoff, F_sample])

    # Convert cutoff frequencies into points on spectrum
    [Low_point, High_point] = map(
        lambda F: F/F_sample * M, [Low_cutoff, High_cutoff])

    # Calculating which frequency has max power.
    maximumFrequency = np.where(
        Spectrum == np.max(Spectrum[Low_point: High_point]))

    return maximumFrequency


def calcFFT(archivo1):
    global figureIndex
    figureIndex+=1
    index = 0
    umbral = 100
    a = 0.90

    sampleRate, data1 = scipy.io.wavfile.read(archivo1)
    preenfasis = np.int16([0 for i in range(len(data1))])
    print(len(preenfasis))

    for n in range(1,len(data1)):
        if abs(data1[n])>umbral:
            preenfasis[index] = data1[n]
            index=index+1
    preenfasis = [i for i in preenfasis if i != 0]

    for n in range(1,len(preenfasis)):
        preenfasis[n] = data1[n] -a*data1[n-1]
    print(len(preenfasis))

    n = 480  # group size
    m = 80  # overlap size
    chunks = [preenfasis[i:i+n] for i in range(0, len(preenfasis), n-m)]

    for chunk in chunks:
        hamming = np.hamming(len(chunk))
        for i in range(0,len(chunk)):
            chunk[i] = chunk[i]*hamming[i]

    arrayFFTS = [0 for i in range(len(chunks))]
    for i in range(len(chunks)):
        arrayFFTS[i] = fft(chunks[i])

    averageFreq =np.double([0 for i in range(n)])
    for i in range(n):
        average = 0
        for c in range(len(chunks)-3):
            average = average + chunks[c][i]
        average = average/len(chunks)-3
        averageFreq[i] = average 

    duration = len(averageFreq)/sampleRate
    time = np.arange(0,duration,1/sampleRate) #time vector
    sp = fftshift(fft(averageFreq))
    freq = fftshift(fftfreq(averageFreq.shape[-1]))
    plt.plot(freq, sp.real, freq, sp.imag)
    #Spectrum = sf.rfft(data1, n=M)
    #maxFrequency(data1, 1/sampleRate)


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22000
CHUNK = 1025
duracion = 2
dirname = os.path.dirname(__file__)
archivo1 = os.path.join(dirname, "audio1.wav")
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
figureIndex = 1

seleccion = int(
    input("Selecciona: \n 1. Grabar audio \n 2. Utilizar audio1.wav de carpeta \n"))
if seleccion == 1:
    input("Presione Enter para grabar el primer audio...")
    recordAudio(archivo1, stream)

stream.close()
audio.terminate()

menuMain = 1

while menuMain != 0:
    if menuMain == 1:  # Suma
        print("Reproduciendo audios...")
        playAudio("audio1.wav")
        plotAudio(archivo1)
        calcFFT(archivo1)
        #playAudio("suma.wav")
    elif menuMain == 2:  # Resta
        time.sleep(1)

    else:
        print("Ingrese una opcion valida")
    plt.show()
    menuMain = int(input("Menu O.B. \n 1.Hombre\n 2.Mujer\n"))

