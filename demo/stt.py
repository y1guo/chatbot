import whisper
from utils import freeup_vram


def load_whisper(name):
    freeup_vram("model")
    global model
    model = whisper.load_model(name)
    return f"Whisper loaded with the {name} model"


def transcribe(audio):
    global model
    result = model.transcribe(audio)
    return result["text"]
