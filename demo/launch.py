import warnings

warnings.filterwarnings("ignore")

import gradio as gr, os
from stt import load_whisper
from tts import load_tts
from chat import load_gpt, generate_prompt, chat, EXAMPLE_FEW_SHOTS


with gr.Blocks() as demo:
    gr.Markdown("""# <center>Yi's Chatbot</center>""")
    with gr.Row():
        with gr.Column(scale=1):
            gpt_name_radio = gr.Radio(
                [
                    "EleutherAI/gpt-neo-125M",
                    "EleutherAI/gpt-neo-1.3B",
                    "EleutherAI/gpt-neo-2.7B",
                    "EleutherAI/gpt-j-6B",
                    "THUDM/chatglm-6b",
                    "THUDM/chatglm2-6b",
                    "meta-llama/Llama-2-7b-hf",
                    "meta-llama/Llama-2-13b-hf",
                    "meta-llama/Llama-2-7b-chat-hf",
                    "meta-llama/Llama-2-13b-chat-hf",
                ],
                value="meta-llama/Llama-2-7b-chat-hf",
                label="LLM model",
            )
            gpt_dtype_radio = gr.Radio(["int4", "int8", "fp16", "fp32"], value="fp16", label="model dtype")
            gpt_in_use_box = gr.Textbox(label="GPT in use", show_label=False)
            gpt_load_button = gr.Button("Load GPT Model")
            do_sample_checkbox = gr.Checkbox(True, label="Do sample")
            temperature_slider = gr.Slider(0, 1.2, 0.8, label="Temperature")
            top_k_slider = gr.Slider(0, 100, 50, label="Top k")
            top_p_slider = gr.Slider(0, 1, 0.95, label="Top p")
            max_new_tokens_box = gr.Number(50, label="Max new tokens")
            user_name_box = gr.Textbox("Bob", label="User name")
            bot_name_box = gr.Textbox("Alice", label="AI name")
        with gr.Column(scale=3):
            with gr.Accordion("Few shot traning prompt", open=False):
                few_shot_training_prompt_box = gr.Textbox(lines=20, show_label=False)
            clear_history_button = gr.Button("Clear History")
            chat_history_box = gr.Chatbot(
                label="Chat History",
                height=800,
                bubble_full_width=False,
                avatar_images=(
                    os.path.join(os.path.dirname(__file__), "assets", "user-avatar.svg"),
                    os.path.join(os.path.dirname(__file__), "assets", "chatbot-avatar.jpg"),
                ),
            )
            chat_input_box = gr.Textbox(
                show_label=False, placeholder="What's the ultimate answer to life, the universe, and everything?"
            )
            with gr.Accordion("Debug", open=False):
                debug_info_box = gr.Textbox(lines=20, show_label=False)
        with gr.Column(scale=1):
            sst_model_radio = gr.Radio(
                ["tiny", "base", "small", "medium", "large"], value="tiny", label="Speech-to-Text model"
            )
            sst_load_box = gr.Textbox(show_label=False)
            sst_load_button = gr.Button("Load Speech-to-Text Model")
            chat_input_audio = gr.Audio(label="Microphone", source="microphone", type="filepath")
            tts_model_radio = gr.Radio(["Linus Tech Tips"], value="Linus Tech Tips", label="Text-to-Speech model")
            tts_load_box = gr.Textbox(show_label=False)
            tts_load_button = gr.Button("Load Text-to-Speech Model")
            response_volume = gr.Slider(0, 100, 67, label="Volume", step=1)
            chat_response_audio = gr.Audio(autoplay=True)
            example_few_shots_dropdown = gr.Dropdown(
                ["", "GPT example 1", "GPT example 2", "猫娘"], label="Select Few Shots Traning Examples"
            )

    gpt_load_button.click(load_gpt, inputs=[gpt_name_radio, gpt_dtype_radio], outputs=gpt_in_use_box)
    clear_history_button.click(lambda: ([], []), None, [chat_history_box, chat_history_box])
    chat_input_box.submit(
        chat,
        [
            example_few_shots_dropdown,
            chat_history_box,
            chat_input_box,
            do_sample_checkbox,
            temperature_slider,
            top_k_slider,
            top_p_slider,
            max_new_tokens_box,
            user_name_box,
            bot_name_box,
            response_volume,
        ],
        [
            chat_history_box,
            chat_input_box,
            chat_response_audio,
            debug_info_box,
        ],
    )
    sst_load_button.click(load_whisper, sst_model_radio, sst_load_box)
    tts_load_button.click(load_tts, tts_model_radio, tts_load_box)
    chat_input_audio.stop_recording(
        chat,
        [
            example_few_shots_dropdown,
            chat_history_box,
            chat_input_audio,
            do_sample_checkbox,
            temperature_slider,
            top_k_slider,
            top_p_slider,
            max_new_tokens_box,
            user_name_box,
            bot_name_box,
            response_volume,
        ],
        [
            chat_history_box,
            chat_input_audio,
            chat_response_audio,
            debug_info_box,
        ],
    )
    example_few_shots_dropdown.change(
        lambda _: generate_prompt(EXAMPLE_FEW_SHOTS[_], "<usr>", "<bot>"),
        example_few_shots_dropdown,
        few_shot_training_prompt_box,
    )

demo.launch(share=True)
