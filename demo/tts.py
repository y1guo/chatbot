import torch, os
import numpy as np
from vits2.utils.task import load_checkpoint
from vits2.utils.model import intersperse
from vits2.utils.hparams import get_hparams_from_file
from vits2.model.models import SynthesizerTrn
from vits2.text.symbols import symbols
from vits2.text import text_to_sequence, PAD_ID
from commons import DEVICE_AUX, freeup_vram
from typing import cast
from numpy.typing import NDArray


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
    global net_g, hps
    freeup_vram("net_g", "hps")
    file_dir = os.path.dirname(__file__)
    hps = get_hparams_from_file(os.path.join(file_dir, "vits2/datasets/custom_base/config.yaml"))
    filter_length = hps.data.n_mels if hps.data.use_mel else hps.data.n_fft // 2 + 1  # type: ignore
    segment_size = hps.train.segment_size // hps.data.hop_length  # type: ignore
    net_g = SynthesizerTrn(len(symbols), filter_length, segment_size, **hps.model).to(DEVICE_AUX)  # type: ignore
    net_g.eval()
    # find the lastest checkpoint
    num = 0
    for file in os.listdir("vits2/datasets/custom_base/logs"):
        if file.startswith("G_") and file.endswith(".pth"):
            num = max(num, int(file.split("G_")[1].split(".pth")[0]))
    model_name = os.path.join(file_dir, f"vits2/datasets/custom_base/logs/G_{num}.pth")
    load_checkpoint(model_name, net_g, None)
    return f"Loaded {model_name}"


def get_text(text: str, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners, language=hps.data.language)
    # text_norm = cleaned_text_to_sequence(text)
    if hps.data.add_blank:
        text_norm = intersperse(text_norm, PAD_ID)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def speak(text: str):
    """Generate audio from text.

    Parameters
    ----------
    text : str
        Text to be synthesized

    Returns
    -------
    1-d array
        Audio waveform
    """
    wav = np.zeros(20000, dtype=np.float32)
    try:
        stn_tst = get_text(text, hps)
        with torch.no_grad():
            x_tst = stn_tst.to(DEVICE_AUX)[None, :]
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(DEVICE_AUX)
            wav = cast(
                NDArray[np.float32],
                (
                    net_g.infer(x_tst, x_tst_lengths, noise_scale=0.667, noise_scale_w=0.8, length_scale=1)[0][0, 0]  # type: ignore
                    .data.to("cpu")
                    .float()
                    .numpy()
                ),
            )
    except:
        print(f"[System Error]: Failed to generate audio for '{text}'.")
    return wav
