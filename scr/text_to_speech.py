from gtts import gTTS
from io import BytesIO

def speak_text(text, lang="en"):
    if not text:
        text = "Hello, this is a test speech."
    tts = gTTS(text=text, lang=lang)
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    return mp3_fp
