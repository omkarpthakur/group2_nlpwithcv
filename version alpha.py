from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import speech_recognition as sr
from gtts import gTTS
import os

model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Function to answer a question
def answer_question(question):
    inputs = tokenizer(question, return_tensors="pt")
    start_positions, end_positions = model(**inputs)
    answer = tokenizer.decode(inputs.input_ids[0][start_positions[0]:end_positions[0] + 1])
    return answer

# Function to listen for the callout word
def listen_for_callout():
    print("Say the callout word 'Hey Friday'")
    speak("Say the callout word 'Hey Friday'")

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for the callout word 'Hey Friday'...")
        audio = r.listen(source)

    try:
        callout = r.recognize_google(audio)
        if "hey friday" in callout.lower():
            return True
        return False
    except sr.UnknownValueError:
        return False
    except sr.RequestError:
        print("Could not request results; check your mic sir.")
        return False

# Function to convert text to speech
def speak(text, output_dir="C:/Users/omkar/OneDrive/AMENA/output"):
    output_file = os.path.join(output_dir, "output.mp3")
    tts = gTTS(text)
    tts.save(output_file)
    os.system("mpg123 " + output_file)

# Main conversation loop
while True:
    if listen_for_callout():
        while True:
            with sr.Microphone() as source:
                print("Speak your question (or say 'exit' to quit): ")
                audio = sr.listen(source)

            try:
                question = sr.recognize_google(audio)
                print("Question:", question)
                if question.lower() == 'exit':
                    speak("Goodbye!")
                    break

                answer = answer_question(question)
                print("Answer:", answer)
                speak(answer)

            except sr.UnknownValueError:
                print("Sorry, I couldn't understand that.")
            except sr.RequestError:
                print("Could not request results; check your internet connection.")

        break
