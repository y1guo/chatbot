import whisper, opencc
from utils import DEVICE, freeup_vram


converter = opencc.OpenCC("t2s.json")


def load_whisper(name: str):
    """Load Whisper model.

    Parameters
    ----------
    name : str
        Model name. Example: "small"

    Returns
    -------
    str
        Status message
    """
    global model
    freeup_vram("model")
    model = whisper.load_model(name, device=DEVICE)
    return f"Whisper loaded with the {name} model"


def transcribe(audio: str):
    """Transcribe audio. If Chinese is detected, return simplified Chinese.

    Parameters
    ----------
    audio : str
        Audio file path

    Returns
    -------
    str
        Transcribed text
    """
    global model
    result = model.transcribe(audio)
    return converter.convert(result["text"])
