import gradio as gr
import numpy as np
from datetime import datetime
from stt import load_whisper, transcribe
from tts import speak
from chat import load_gpt, chat_regulator, chat_response, generate_prompt, example_few_shots


def chat(
    example_few_shots_dropdown,
    chat_history_list,
    chat_input,
    do_sample,
    temperature,
    top_k,
    top_p,
    max_new_tokens,
    user_name,
    bot_name,
):
    print(chat_input)
    # detect audio filepath if it's audio
    if chat_input.startswith("/tmp/"):
        chat_input = transcribe(chat_input)

    # prepare prompt including few-shot and chat history
    # regulate each term every iteration since there might be '<br>' added into them.
    chat_history_list = [
        (chat_regulator(user_text), chat_regulator(bot_text))
        for (user_text, bot_text) in chat_history_list
    ]

    response, chat_history_list, text = chat_response(
        chat_input,
        chat_history_list,
        example_few_shots_dropdown,
        max_new_tokens,
        do_sample,
        temperature,
        top_k,
        top_p,
        user_name,
        bot_name,
    )

    # output to .wav using tts
    wav = speak(response)
    wav_int16 = (np.array(wav) * 32767).astype(np.int16)

    # monitor the conversation in the command line
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("log.txt", "a", encoding="utf8") as fout:
        fout.write(f"{now} <usr>: {chat_input}\n{now} <bot>: {response}\n")

    return (
        chat_history_list,
        None,
        (22050, wav_int16),
        str([text]) + "\n" + str(chat_history_list),
    )


with gr.Blocks() as demo:
    gr.Markdown("""# <center>人  工  智  障</center>""")
    with gr.Row():
        with gr.Column(scale=1):
            gpt_name_radio = gr.Radio(
                [
                    "EleutherAI/gpt-neo-125M",
                    "EleutherAI/gpt-neo-1.3B",
                    "EleutherAI/gpt-neo-2.7B",
                    "EleutherAI/gpt-j-6B",
                    "THUDM/chatglm-6b",
                ],
                value="EleutherAI/gpt-j-6B",
                label="GPT model",
            )
            gpt_dtype_radio = gr.Radio(
                ["int8", "fp16", "fp32"], value="int8", label="GPT dtype"
            )
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
                few_shot_training_prompt_box = gr.Textbox(show_label=False)
            clear_history_button = gr.Button("Clear History")
            chat_history_box = gr.Chatbot(label="History")
            chat_input_box = gr.Textbox(
                show_label=False, placeholder="Do you think I'm annoying?"
            )
            with gr.Accordion("Debug", open=False):
                debug_info_box = gr.Textbox(show_label=False)
        with gr.Column(scale=1):
            sst_model_radio = gr.Radio(
                ["tiny", "base", "small"], value="tiny", label="Speech-to-Text model"
            )
            sst_load_box = gr.Textbox(show_label=False)
            sst_load_button = gr.Button("Load Speech-to-Text Model")
            chat_input_audio = gr.Audio(
                label="Microphone", source="microphone", type="filepath"
            )
            chat_response_audio = gr.Audio()
            example_few_shots_dropdown = gr.Dropdown(["猫娘","GPT example 1", "GPT example 2"], label="Select Few Shots Traning Examples")

    gpt_load_button.click(
        load_gpt, inputs=[gpt_name_radio, gpt_dtype_radio], outputs=gpt_in_use_box
    )
    clear_history_button.click(
        lambda: ([], []), None, [chat_history_box, chat_history_box]
    )
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
        ],
        [
            chat_history_box,
            chat_input_box,
            chat_response_audio,
            debug_info_box,
        ],
    )
    sst_load_button.click(load_whisper, sst_model_radio, sst_load_box)
    chat_input_audio.change(
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
        ],
        [
            chat_history_box,
            chat_input_audio,
            chat_response_audio,
            debug_info_box,
        ],
    )
    example_few_shots_dropdown.change(lambda _:generate_prompt(example_few_shots[_], "<usr>", "<bot>"), example_few_shots_dropdown, few_shot_training_prompt_box)

demo.launch(share=True)
