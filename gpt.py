from bert import wrap_text_dynamic , answer_question
import language_tool_python
tool = language_tool_python.LanguageTool('en-US')
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def correct_grammar(sentence):
    # Load pre-trained model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Tokenize input sentence
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    minimum = len(sentence)
    maximum = minimum

    # Generate corrected text
    output = model.generate(input_ids,min_length = minimum , max_length=maximum, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    # Decode and return corrected sentence
    corrected_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
    return corrected_sentence

"final function "
def QNA(question,context):
    ques = question
    context = context
    abstract = wrap_text_dynamic(context)
    answer = answer_question(ques,abstract)
    correct_answer = correct_grammar(answer)
    return answer





