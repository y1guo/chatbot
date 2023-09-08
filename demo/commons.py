import torch
import gc


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_AUX = torch.device(f"cuda:{torch.cuda.device_count() - 1}" if torch.cuda.is_available() else "cpu")


# freeup the model and free up GPU RAM
def freeup_vram(*args):
    """Free up GPU RAM.

    Parameters
    ----------
    *args : str
        Variable names to be deleted
    """
    memory_used_before = torch.cuda.memory_reserved(0) / 1024**3
    for var in args:
        if var in globals():
            try:
                globals()[var].to("cpu")
            except:
                pass
            del globals()[var]
            print(f"'{var}' deleted from memory.")
    gc.collect()
    torch.cuda.empty_cache()
    memory_used_after = torch.cuda.memory_reserved(0) / 1024**3
    print(f"Freed up {memory_used_before - memory_used_after:.1f} GB of VRAM.")
