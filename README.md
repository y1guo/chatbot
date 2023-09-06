# Chatbot

A chatbot demo that allows speech-to-speech or text based chat experience with the LLM. Local deployment. No APIs used.

## Environment

Create a virtual environment, I used conda here

```bash
conda create -n chatbot python=3.10
```

Activate the virtual environment

```bash
conda activate chatbot
```

Install PyTorch

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install ffmpeg if you pytorch does not come with ffmpeg

```bash
conda install ffmpeg -c pytorch
```

Install other dependencies

```bash
pip install gradio transformers TTS openai-whisper 
```

Now, it's sufficient for the chatbot to run with GPT-J and GPT-neo family.

(Optional) Add support for ChatGLM2-6B

```bash
pip install transformers==4.30.2 cpm_kernels mdtex2html sentencepiece accelerate
```

(Optional) Allow loading GPT-J-6B in ```int8```, which reduces the VRAM requirement to 6GB. 

```bash
pip install bitsandbytes accelerate
```

### Note

-   In WSL, I also need to symlink ```libcuda.so``` to the environment library. Otherwise it seems ```bitsandbytes``` could not find it.

    Example:

    ```bash
    ln -s /usr/lib/wsl/lib/libcuda.so /home/USERNAME/miniconda3/envs/chatbot/lib/libcuda.so
    ```

## Launch the Demo

```bash
python demo/demo.py
```

Then open `https://localhost:7860` in the browser, or the link provided after the server fully started.

## To-Do

-   Replace `whisper` with `whisperX` to speed up speech recognition.
