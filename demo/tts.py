from TTS.api import TTS
from utils import DEVICE, freeup_vram


def load_tts(model_name: str):
    """Load TTS model.

    Parameters
    ----------
    model_name : str
        Model name. Example: "en/vctk/vits"

    Returns
    -------
    str
        Status message
    """
    global tts
    freeup_vram("tts")
    tts = TTS(model_name=f"tts_models/{model_name}").to(DEVICE)
    return f"Loaded {model_name}"


def speak(text: str):
    """Generate audio from text.

    Parameters
    ----------
    text : str
        Text to be synthesized

    Returns
    -------
    list
        Audio waveform
    """
    # # selected speakers: p243, p259, p263, p270, p306
    speaker = "p306" if tts.speakers and "p306" in tts.speakers else None
    try:
        wav = tts.tts(text=text, speaker=speaker)  # type: ignore
    except:
        wav = [0 for _ in range(20000)]
        print(f"[System Error]: Failed to generate audio for '{text}'.")
    return wav
