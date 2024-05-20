import speech_recognition as sr
import os
import webbrowser
import datetime
import random
import numpy as np
from gpt import QNA
from cvonnlp import Wosc

chatStr = ""


def say(text):
    os.system(f'say "{text}"')

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        # r.pause_threshold =  0.6
        audio = r.listen(source)
        try:
            print("Recognizing...")
            query = r.recognize_google(audio, language="en-in")
            print(f"User said: {query}")
            return query
        except Exception as e:
            return "Some Error Occurred. Sorry from amena"


Run = True
while Run == True :
    print('Welcome to Amena')
    say("heyy amena")
    while True:
        print("Listening...")
        query = takeCommand()
        # todo: Add more sites
        sites = [["youtube", "https://www.youtube.com"], ["wikipedia", "https://www.wikipedia.com"], ["google", "https://www.google.com"],]
        for site in sites:
            if f"Open {site[0]}".lower() in query.lower():
                say(f"Opening {site[0]} sir...")
                webbrowser.open(site[1])
        if "the time" in query:

            hour = datetime.datetime.now().strftime("%H")
            min = datetime.datetime.now().strftime("%M")
            say(f" time is {hour} : {min} minutes")

        elif "open facetime".lower() in query.lower():
            os.system(f"open /System/Applications/FaceTime.app")

        elif "open pass".lower() in query.lower():
            os.system(f"open /Applications/Passky.app")



        elif "using artificial intelligence tell me".lower() in query.lower():

            q = query.lower().replace("using artificial intelligence tell me", "").strip()

            print("what is context")

            C = takeCommand()

            c = C.lower()

            print("Context:", c)

            print('Amena : ',QNA(question=q, context=c))

        elif "use vision".lower() in query.lower():
            print("shall we start")
            Q = takeCommand()
            q = Q.lower()
            print('Amena : ',Wosc(inp=q))



        elif "good night ".lower() in query.lower():
            print("amena :good night")
            Run = False

        elif "reset chat".lower() in query.lower():
            chatStr = ""

        elif "screen".lower() in query.lower():
            pass


        else:
            print("Chatting...")
