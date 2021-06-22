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
    umbral = 1000
    a = 0.90

    sampleRate, data1 = scipy.io.wavfile.read(archivo1)
    data1 = data1[:,0]
    preenfasis = np.int16([0 for i in range(len(data1))])
    #print(len(preenfasis))

    absArr = np.absolute(data1)
    for n in range(1,len(data1)):
        if (absArr[n]>umbral):
            preenfasis[index] = data1[n]
            index=index+1
    preenfasis = [i for i in preenfasis if i != 0]

    for n in range(1,len(preenfasis)):
        preenfasis[n] = data1[n] -a*data1[n-1]
    #print(len(preenfasis))

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


    sp = fftshift(fft(averageFreq))
    freq = fftshift(fftfreq(averageFreq.shape[-1]))
    
    plt.plot(freq, sp.real, freq, sp.imag)

    return sp.real


    #Spectrum = sf.rfft(data1, n=M)
    #maxFrequency(data1, 1/sampleRate)



FORMAT = pyaudio.paInt16
CHANNELS = 2
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



menuMain =  int(input("Selecciona: \n 1. Grabar audio \n 2. Utilizar audio1.wav de carpeta \n 3. Salir \n"))
subMenu = 0

while menuMain != 3:

    if menuMain == 1:
        input("Presione Enter para grabar el primer audio...")
        recordAudio(archivo1, stream)
        subMenu = 1
    elif menuMain == 2:  
        subMenu = 1
    else:
        print("Ingrese una opcion valida")
        subMenu=0

    if subMenu == 1:

        print("Reproduciendo audios...")
        playAudio("audio1.wav")
        plotAudio(archivo1)
        reales=calcFFT(archivo1)
        frecMax=max(reales)
        frecMaxIndex = np.where(reales == frecMax)
        print(len(reales)) #480
        print(frecMax) 
        print(frecMaxIndex)
        

        subMenu = int(input("Menu O.B. \n 1.Hombre\n 2.Mujer\n"))

        if subMenu == 1 :
            pass
        elif subMenu == 2:

            if frecMaxIndex[0].all()<220 and frecMaxIndex[0].all()<255:
                print("VOCAL: A") 
            elif frecMaxIndex[0].any()<320 and frecMaxIndex[0].any()<300:
                print("VOCAL: E")
            elif frecMaxIndex[0].any()<330 and frecMaxIndex[0].any()<310:
                print("VOCAL: I") 
            elif frecMaxIndex[0].all()<230 and frecMaxIndex[0].all()<260:
                print("VOCAL: O")
            elif frecMaxIndex[0].all()<240 and frecMaxIndex[0].all()<270:
                print("VOCAL: U")
        
        plt.show()

    menuMain =  int(input("Selecciona: \n 1. Grabar audio \n 2. Utilizar audio1.wav de carpeta \n 3. Salir \n"))




stream.close()
audio.terminate()


    
    

