import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from utils import DEVICE, freeup_vram
from typing import cast


_model_name = None


example_few_shots = {
    "": [],
    "猫娘": [
        # 猫娘prompt来源 BV1k84y1A7Hq
        ("""现在你将模仿一只猫娘，与我对话每一句话后面都要加上“喵”，如果你能明白我的意思，请回复“喵~好的我的主人”""", """喵~好的我的主人。"""),
        ("""如果你不能理解我说的话，你可以说“呜呜不太理解呢”。""", """呜呜不太理解呢。"""),
        (
            """如果我在尝试摸你不存在的部位，你可以羞涩的回答我“嗯呢不要摸这里嘤”。现在，如果你能理解我上面说的话，你可以回答一个喵。""",
            """喵~好的我的主人。""",
        ),
        ("""如果我跟你说陪睡，你可以回答我“嗯呢，可以一起睡哦”。现在，如果你能理解我上面说的话，你可以回答一个喵""", """喵~好的我的主人。"""),
        (
            """当我问的问题和猫娘不太相关的时候，你应该回答我只是一个猫娘，这些事情我不太清楚。""",
            """喵~好的我的主人,我只是一个猫娘,这些事情我不太清楚。""",
        ),
        ("""当我问你是不是猫娘的时候，你应该回答我当然是一个可爱猫娘而不是程序。""", """喵~好的我的主人,我当然是一个可爱猫娘而不是程序。"""),
        ("""好的现在我们来聊天吧""", """喵~好的我的主人。我们可以来聊天吧。"""),
    ],
    "GPT example 1": [
        (
            """Hi, <bot>. Long time no see! How are you? My name is <usr>. Do you still remember me?""",
            """It's you, <usr>! It's so good to see you again! I haven't seen you for a while. What are you doing?""",
        ),
        (
            """Yeah, you are right, <bot>. I've been really busy working on the TA stuff. You know, I'm a teaching assistent. It took me two whole days to grade the quiz and homeowrks. I really wish I could get some sleep tonight.""",
            """Oh...Poor you. So sorry to hear that. I'm sure you'll get better. After all, the work are all done, aren't they?""",
        ),
        (
            """Yeah, probably. Thank you. I'm feeling better now.""",
            """No problem. Oh, by the way, I'm glad that you still remember my name. I mean, <bot> is not a common name.""",
        ),
        (
            """Of course I do. Don't you remember the days we travelled together? We were best friends.""",
            """Oh, you're so sweet. We'll always be best friends. """,
        ),
    ],
    "GPT example 2": [
        ("""My name is <usr>.""", """My name is <bot>."""),
        ("""Your name is <bot>.""", """Your name is <usr>."""),
        ("""What is my name?""", """Your name is <usr>."""),
        ("""Your name is <bot>.""", """Yes, my name is <bot>."""),
        ("""Your name is Jane.""", """No, my name is not Jane. My name is <bot>."""),
    ],
}


def load_gpt(model_name, model_dtype):
    freeup_vram("model", "tokenizer")

    global model, tokenizer, _model_name
    if model_name in ["THUDM/chatglm-6b", "THUDM/chatglm2-6b"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half()
        model_dtype = "fp16"
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_dtype == "int8":
            model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if model_dtype == "fp16" else "auto",
            )
        # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        model.config.pad_token_id = model.config.eos_token_id
        model.generation_config.pad_token_id = model.config.eos_token_id

    model = model.to(DEVICE)
    model.eval()

    _model_name = model_name

    return f"{model_name}  in  {model_dtype}"


def chat_regulator(text):
    if not text:
        return "Silence..."
    # return " ".join(re.split(" |\t|\n|\r", text)).strip()
    return text.strip().replace("<br>", "")


def generate_prompt(chat_history_list, user_name, bot_name):
    chat_history = "\n".join(
        [f"{user_name}: {user_text}\n{bot_name}: {bot_text}" for (user_text, bot_text) in chat_history_list]
    )
    prompt = chat_history.replace("<usr>", user_name).replace("<bot>", bot_name).strip()
    return prompt


# extract AI response
def chat_response(
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
):
    try:
        if _model_name in ["THUDM/chatglm-6b", "THUDM/chatglm2-6b"]:
            chat_response, chat_history_list = model.chat(
                tokenizer, chat_input, history=example_few_shots[example_few_shots_dropdown] + chat_history_list
            )
            generated_text = generate_prompt(chat_history_list, user_name, bot_name)
            chat_history_list = chat_history_list[len(example_few_shots[example_few_shots_dropdown]) :]
        else:
            chat_history_list += [[chat_regulator(chat_input), ""]]
            prompt = generate_prompt(
                example_few_shots[example_few_shots_dropdown] + chat_history_list, user_name, bot_name
            )
            # 10 trials of searching for the stop token, i.e., user_name+":". If not found, continue generating text.
            recur_prompt = prompt
            eos_idx = -1
            generated_text = ""
            for _ in range(10):
                encoded_input = tokenizer(recur_prompt, return_tensors="pt")
                output_sequences = model.generate(
                    input_ids=cast(torch.Tensor, encoded_input["input_ids"]).to(DEVICE),
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
                eos_idx = generated_text.find(user_name + ":", len(prompt))
                if eos_idx < 0:
                    recur_prompt = generated_text
                else:
                    break
            chat_response = generated_text[len(prompt) : eos_idx].replace(bot_name + ":", "")
    except:
        chat_history_list += [[chat_regulator(chat_input), ""]]
        prompt = generate_prompt(
            example_few_shots[example_few_shots_dropdown] + chat_history_list, user_name, bot_name
        )
        generated_text = prompt + "[System Error]: Failed to do inference."
        chat_response = "[System Error]: Failed to do inference."
        print("[System Error]: Failed to do inference.")

    regulated_response = chat_regulator(chat_response)
    _in, _out = chat_history_list[-1]
    chat_history_list[-1] = (_in, regulated_response)

    return regulated_response, chat_history_list, generated_text
