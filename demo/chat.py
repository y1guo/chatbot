import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import freeup_vram


def load_gpt(model_name, model_dtype):
    if model_name.endswith("6B"):
        model_dtype = "int8"
    # elif model_name.endswidth('2.7B') and model_dtype == 'fp32':
    #     model_dtype = 'fp16'

    freeup_vram("model", "tokenizer")

    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_dtype == "int8":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", load_in_8bit=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if model_dtype == "fp16" else "auto",
        )

    model.eval()

    # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id

    return f"{model_name}  in  {model_dtype}"


def chat_regulator(text):
    if not text:
        return "Silence..."
    # return " ".join(re.split(" |\t|\n|\r", text)).strip()
    return text.strip().replace("<br>", "")


# extract AI response
def chat_response(
    prompt, max_new_tokens, do_sample, temperature, top_k, top_p, user_name, bot_name
):
    try:
        # 10 trials of searching for the stop token, i.e., user_name+":". If not found, continue generating text.
        recur_prompt = prompt
        for _ in range(10):
            encoded_input = tokenizer(recur_prompt, return_tensors="pt")
            output_sequences = model.generate(
                input_ids=encoded_input["input_ids"].cuda(),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            generated_text = tokenizer.decode(
                output_sequences[0], skip_special_tokens=True
            )
            eos_idx = generated_text.find(user_name + ":", len(prompt))
            if eos_idx < 0:
                recur_prompt = generated_text
            else:
                break
    except:
        generated_text = prompt + "[System Error]: Failed to do inference."
        eos_idx = -1
        print("[System Error]: Failed to do inference.")

    chat_response = generated_text[len(prompt) : eos_idx].replace(bot_name + ":", "")
    regulated_response = chat_regulator(chat_response)

    return regulated_response, generated_text
