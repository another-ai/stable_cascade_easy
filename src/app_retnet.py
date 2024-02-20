from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import re

def contains_banned_word(word, banned_words,prompt_chara):
    banned_words_text = banned_words
    if prompt_chara:
        banned_words_text = banned_words_text + ["chibi","eyes","hair","heterochromia","mask","mole"]
    for banned_word_text in banned_words_text:
        if re.search(banned_word_text, word):
            return True
    return False

def replace_word(text, banned_words=[],prompt_chara=False):

    cleaned_words = [word for word in text.split(",") if not contains_banned_word(word, banned_words, prompt_chara)]

    cleaned_text = ",".join(cleaned_words)

    return cleaned_text

def main_def(prompt_input, max_tokens=256, DEVICE="cpu", banned_words=[], prompt_chara=False):
    # if prompt_input == "":
    #   prompt_input = "1woman"

    MODEL_NAME = "isek-ai/SDPrompt-RetNet-v2-beta"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    ).to(DEVICE)

    streamer = TextStreamer(tokenizer)

    prompt_input = "<s>"+prompt_input

    inputs = tokenizer(prompt_input, return_tensors="pt", add_special_tokens=False)["input_ids"]

    print(f"Token={max_tokens}")
    token_ = model.generate(
        inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        top_p=0.9,
        top_k=20,
        temperature=0.9,
        streamer=streamer,
        )
    generated_text = tokenizer.decode(token_[0], skip_special_tokens=True).strip()
    if len(banned_words) > 0:
        generated_text = replace_word(generated_text, banned_words, prompt_chara)
    generated_text = generated_text.replace("<s>", "").replace("</s>", "").replace(",,,",",").replace(",,", ",").replace(", ", ",").replace(" ", "_")
    print(generated_text)
    return generated_text

if __name__ == "__main__":
    main_def("")