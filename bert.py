import wikipedia
import textwrap
import shutil
from transformers import BertForQuestionAnswering
import torch
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')



# finding the dat
def context(text):
    try:
        article_summary = wikipedia.summary(text)
        full_article_text = wikipedia.page(text).content
    except wikipedia.exceptions.DisambiguationError as e:
        summaries = []
        full_texts = []
        for option in e.options:
            try:
                summaries.append(wikipedia.summary(option))
                full_texts.append(wikipedia.page(option).content)
            except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
                pass
        if not summaries:
            return "No matching Wikipedia pages found for the given text."
        else:
            #
            return "\n".join(summaries) + "\n\n" + "\n\n".join(full_texts)
    except wikipedia.exceptions.PageError:
        #
        return "No matching Wikipedia page found for the given text."


    return article_summary + "\n" + full_article_text

"""print(context("Python"))
"""

def wrap_text_dynamic(text):
    Text = context(text)
    """
    Wrap the given text based on the width of the terminal window.

    Args:
    text (str): The text to wrap.

    Returns:
    str: The wrapped text.
    """
    # Get the width of the terminal window
    terminal_width = shutil.get_terminal_size().columns

    # Wrap the text based on the terminal width
    wrapper = textwrap.TextWrapper(width=terminal_width)
    return wrapper.fill(Text)
def wrap_text_dynamic(text):
    Text = context(text)
    """
    Wrap the given text based on the width of the terminal window.

    Args:
    text (str): The text to wrap.

    Returns:
    str: The wrapped text.
    """
    # Get the width of the terminal window
    terminal_width = shutil.get_terminal_size().columns

    # Wrap the text based on the terminal width
    wrapper = textwrap.TextWrapper(width=terminal_width)
    return wrapper.fill(Text)


def extract_keywords(text, model_name='bert-base-uncased', top_n=5):

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state

    pooled_output = torch.mean(last_hidden_states, dim=1)

    features = pooled_output.numpy()

    feature_texts = [tokenizer.decode(token.squeeze(), skip_special_tokens=True) for token in inputs['input_ids']]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(feature_texts)
    tfidf = X.toarray()

    keywords = [vectorizer.get_feature_names_out()[np.argmax(feature)] for feature in tfidf]

    # Join keywords into a single string
    keywords_string = ' '.join(keywords)

    return keywords_string

# ut3
"""text = str(input("give text"))
keywords = extract_keywords(text)
print("Keywords:", keywords)"""



#bert



import torch
import pandas as pd
from transformers import BertForQuestionAnswering
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
from transformers import BertTokenizer
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

