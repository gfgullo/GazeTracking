from threading import Thread
from playsound import playsound

class Audio:

    def play(self, audio_file):
        thr = Thread(target=self._play, args=(), kwargs={"sound_file":"resources/"+audio_file})
        thr.start()

    def _play(self, sound_file=None):
        playsound(sound_file)


