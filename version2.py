from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import speech_recognition as sr
from gtts import gTTS
import os
from time import localtime

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

"speach to text"
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak something...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand what you said.")
    except sr.RequestError as e:
        print("Sorry, couldn't request results from Google Speech Recognition service; {0}".format(e))
        return None
"text to speach"
def speak(text, output_dir="C:/Users/omkar/OneDrive/AMENA/output"):
    lc = str(localtime())+".mp3"
    output_file = os.path.join(output_dir, lc)
    tts = gTTS(text)
    tts.save(output_file)
    os.system("mpg123 " + output_file)

"language model"
def answer_question():
    print("Ask me a question:")
    question = speech_to_text()
    if not question:
        return

    print("User asked:", question)
    print("Please provide more context:")
    context = speech_to_text()
    if not context:
        return

    print("User context:", context)

    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_logits = outputs.start_logits
    answer_end_logits = outputs.end_logits

    answer_start = torch.argmax(answer_start_logits)
    answer_end = torch.argmax(answer_end_logits) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    print("Answer:", answer)
    speak(answer)

run = True
while run:
    answer_question()

