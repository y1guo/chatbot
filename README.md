# chatbot

## Working platform

Windows WSL.

## Required packages

```
conda create -n chatbot -c conda-forge python=3.10 numpy=1.22.4 pytorch=1.13.1 gradio transformers accelerate ffmpeg ffmpeg-python
conda activate chatbot
pip3 install bitsandbytes TTS openai-whipser
```

```python=3.10, numpy=1.22.4, pytorch=1.13.1``` is required by ```TTS```.

```ffmpeg, ffmpeg-python``` required by ```openai-whisper```.

```bitsandbytes``` is used for loading GPT-J-6B in ```int8```, in which it only takes 6GB of VRAM and can be loaded into GPUs that's other than X090.

In WSL, I also need to symlink ```libcuda.so``` to the environment library seems ```bitsandbytes``` couldn't find it.

```
ln -s /usr/lib/wsl/lib/libcuda.so /home/USERNAME/miniconda3/envs/chatbot/lib/libcuda.so
```
