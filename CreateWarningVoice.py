from gtts import gTTS


def createVoice(text,filename):

    voice=gTTS(text,lang="en")

    voice.save(filename)


createVoice("do not cover camera","BlockedCameraWarning.mp3")