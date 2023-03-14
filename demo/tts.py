from TTS.api import TTS
from utils import freeup_vram


# loading the text-to-speech model
freeup_vram("tts")

tts = TTS(
    model_name="tts_models/en/vctk/vits", gpu=True
)  # speaker: p243, p259, p263, p270, p306


def speak(text):
    try:
        wav = tts.tts(text=text, speaker="p306")
    except:
        wav = [0 for _ in range(20000)]
        print(f"[System Error]: Failed to generate audio for '{text}'.")
    return wav
