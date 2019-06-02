# You need to install pyaudio to run this example
# pip install pyaudio

# When using a microphone, the AudioSource `input` parameter would be
# initialised as a queue. The pyaudio stream would be continuosly adding
# recordings to the queue, and the websocket client would be sending the
# recordings to the speech to text service

from __future__ import print_function
import pyaudio
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
from threading import Thread
import json

try:
    from Queue import Queue, Full
except ImportError:
    from queue import Queue, Full

###############################################
#### Initalize queue to store the recordings ##
###############################################
CHUNK = 1024
# Note: It will discard if the websocket client can't consumme fast enough
# So, increase the max size as per your choice
BUF_MAX_SIZE = CHUNK * 10
# Buffer to store audio
q = Queue(maxsize=int(round(BUF_MAX_SIZE / CHUNK)))

# Create an instance of AudioSource
audio_source = AudioSource(q, True, True)

res_text = 'NULL'
res_speaker = 'NULL'

###############################################
#### Prepare Speech to Text Service ########
###############################################

# initialize speech to text service
speech_to_text = SpeechToTextV1(
    iam_apikey='nhtmrfU38k-4vrSmUmrnJTiMvjQwXCqKlBcxUhLNIt6p',
    url='https://gateway-wdc.watsonplatform.net/speech-to-text/api')


# define callback for the speech to text service
class MyRecognizeCallback(RecognizeCallback):
    def __init__(self):
        RecognizeCallback.__init__(self)

    def on_transcription(self, transcript):
        global res_speaker, res_text

        res_text = transcript[0]['transcript']

    def on_connected(self):
        global res_speaker, res_text

        res_speaker = 'Connection was successful'

    def on_error(self, error):
        global res_speaker, res_text

        res_speaker = 'Error received: {}'.format(error)

    def on_inactivity_timeout(self, error):
        global res_speaker, res_text

        res_speaker = 'Inactivity timeout: {}'.format(error)

    def on_listening(self):
        global res_speaker, res_text

        res_speaker = 'Service is listening'

    def on_hypothesis(self, hypothesis):
        pass

    def on_data(self, data):
        global res_speaker, res_text

        if 'results' in data.keys():
            res_text = data['results'][0]['alternatives'][0]['transcript']
        elif 'speaker_labels' in data.keys():
            res_speaker = str(data['speaker_labels'][0]['speaker'])

    def on_close(self):
        global res_speaker, res_text

        res_speaker = "Connection closed"


# this function will initiate the recognize service and pass in the AudioSource
def recognize_using_weboscket(*args):
    mycallback = MyRecognizeCallback()
    speech_to_text.recognize_using_websocket(audio=audio_source,
                                             content_type='audio/l16; rate=44100',
                                             recognize_callback=mycallback,
                                             speaker_labels=True,
                                             interim_results=True)


###############################################
#### Prepare the for recording using Pyaudio ##
###############################################

# Variables for recording the speech
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100


# define callback for pyaudio to store the recording in queue
def pyaudio_callback(in_data, frame_count, time_info, status):
    try:
        q.put(in_data)
    except Full:
        pass  # discard
    return (None, pyaudio.paContinue)


# instantiate pyaudio
audio = pyaudio.PyAudio()
stream = None


def start_recording():
    try:
        # open stream using callback
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=pyaudio_callback,
            start=False
        )

        stream.start_stream()
        recognize_thread = Thread(target=recognize_using_weboscket, args=())
        recognize_thread.start()
        return True
    except:
        return False


def stop_recording():
    try:
        # stop recording
        audio_source.completed_recording()
        stream.stop_stream()
        stream.close()
        audio.terminate()
        return True
    except:
        return False


def get_text():
    global res_text
    return res_text


def get_speaker():
    global res_speaker
    return res_speaker


import time
if __name__ == '__main__':
    start_recording()
    while True:
        print(get_speaker() + ', ' + get_text())
        time.sleep(0.5)