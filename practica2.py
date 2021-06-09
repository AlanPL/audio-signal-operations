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

def plotAudios(archivo1, archivo2):
    sampleRate, audioBuffer1 = scipy.io.wavfile.read(archivo1)
    sampleRate, audioBuffer2 = scipy.io.wavfile.read(archivo2)
    print('longitud del arreglo de datos: ',len(audioBuffer1))
    duration = len(audioBuffer1)/sampleRate
    time = np.arange(0,duration,1/sampleRate) #time vector
    originalPlot = plt.figure(1)
    plt.plot(time,audioBuffer1, label="Audio 1")
    plt.plot(time,audioBuffer2, label="Audio 2")
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title("Audio 1 & Audio 2")

def plotAudio(archivo):
    global figureIndex
    figureIndex+=1
    sampleRate, audioBuffer = scipy.io.wavfile.read(archivo)
    duration = len(audioBuffer)/sampleRate
    time = np.arange(0,duration,1/sampleRate) #time vector
    figure = plt.figure(figureIndex)
    plt.plot(time,audioBuffer, label="Audio")
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(os.path.basename(archivo))

def opSumaResta(archivo1, archivo2, filename, factor):
    sampleRate, data1 = scipy.io.wavfile.read(archivo1)
    sampleRate, data2 = scipy.io.wavfile.read(archivo2)
    wavesSum = np.int16([0 for i in range(len(data1))])
    for i in range(0, len(data1)):
        ### read 1 frame and the position will updated ###
        wavesSum[i] = data1[i]+factor*data2[i]
    scipy.io.wavfile.write(filename, sampleRate, wavesSum.astype(np.int16))

def opDiezmacion(archivo1, filename, factor):
    sampleRate, data = scipy.io.wavfile.read(archivo1)
    leng = int(len(data)/factor)+ len(data)%factor
    newWave = np.int16([0 for i in range(leng)])
    index = 0
    for i in range(0, len(data)):
        if(i%factor==0):
            newWave[index] = data[i]
            index+=1
    scipy.io.wavfile.write(filename, sampleRate, newWave.astype(np.int16))

def opInterpolacion(archivo1, filename, factor, tipo):
    sampleRate, data = scipy.io.wavfile.read(archivo1)
    leng = int(len(data)*factor)
    newWave = np.int16([0 for i in range(leng)])
    index = 0

    increment = 0
    for i in range(0, leng):
        if(i%factor==0):
            newWave[i] = data[index]
            if(index==len(data)-1):
                sign = 1 if(data[index]<0) else -1
                increment = sign*int(data[index]/factor)
            else:
                sign = 1 if(data[index]<data[index+1]) else -1
                increment = sign*int(abs(data[index+1]-data[index])/factor)
            index+=1
        else:
            if(tipo==0):
                newWave[i] = 0
            elif(tipo==1):
                newWave[i] = data[index-1]
            elif(tipo==2):
                newWave[i] = newWave[i-1]+increment
                
    scipy.io.wavfile.write(filename, sampleRate, newWave.astype(np.int16))

def opDesplazamiento(archivo1, filename, factor):
    sampleRate, data = scipy.io.wavfile.read(archivo1)
    newWave = np.int16([0 for i in range(len(data))])
    index = 0
    if(factor>=0):
        for i in range(factor, len(data)):
            newWave[i] = data[index]
            index+=1
    else:
        index=abs(factor)
        for i in range(0, len(data)+factor-1):
            newWave[i] = data[index]
            index+=1

    scipy.io.wavfile.write(filename, sampleRate, newWave.astype(np.int16))
FORMAT=pyaudio.paInt16
CHANNELS=1
RATE=44100
CHUNK=1025
duracion=4
dirname = os.path.dirname(__file__)
archivo1= os.path.join(dirname, "audio1.wav")
archivo2= os.path.join(dirname, "audio2.wav")
suma= os.path.join(dirname, "suma.wav")
resta= os.path.join(dirname, "resta.wav")
diezmacion= os.path.join(dirname, "diezmacion.wav")
interpolacion= os.path.join(dirname, "interpolacion.wav")
desplazamiento= os.path.join(dirname, "desplazamiento.wav")

audio=pyaudio.PyAudio()
stream=audio.open(format=FORMAT,channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
figureIndex = 1
input("Press Enter to record first audio...")
recordAudio(archivo1, stream)
input("Press Enter to record second audio...")
recordAudio(archivo2, stream)
plotAudios(archivo1, archivo2)
#DETENEMOS GRABACIÓN
stream.close()
audio.terminate()


opSumaResta(archivo1, archivo2, suma, 1)
plotAudio(suma)

opSumaResta(archivo1, archivo2, resta, -1)
plotAudio(resta)

factor = int(input("Ingrese el factor de diezmacion: "))
opDiezmacion(archivo1, diezmacion, factor)
plotAudio(diezmacion)

factor = int(input("Ingrese el factor de interpolacion "))
tipo = int(input("Ingrese el tipo de interpolacion (0=cero, 1=escalon, 2=lineal) "))
opInterpolacion(archivo1, interpolacion, factor, tipo)
plotAudio(interpolacion)


factor = int(input("Ingrese el factor de desplazamiento: "))
opDesplazamiento(archivo1, desplazamiento, factor)
plotAudio(desplazamiento)
plt.show()
