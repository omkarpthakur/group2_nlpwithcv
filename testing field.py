import torch
import pandas as pd
from transformers import BertForQuestionAnswering
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
from transformers import BertTokenizer
from bert import wrap_text_dynamic
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

import torch
from transformers import BertForQuestionAnswering, BertTokenizer

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # Calculate the length of the abstract in tokens
    abstract_length = len(tokenizer.encode(answer_text))

    # Calculate the maximum token length based on the abstract length
    max_length = min(512, abstract_length + len(tokenizer.encode(question)))

    # Tokenize question and answer_text with the dynamically calculated max_length
    inputs = tokenizer(question, answer_text, return_tensors='pt', max_length=max_length, truncation=True)

    # Retrieve the input_ids and token_type_ids
    input_ids = inputs.input_ids
    token_type_ids = inputs.token_type_ids

    # Run the model
    outputs = model(input_ids, token_type_ids=token_type_ids)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    return ('Answer: "' + answer + '"')
abstract = wrap_text_dynamic("python programing")
ques = 'what is python'
print(answer_question(ques ,abstract ))

