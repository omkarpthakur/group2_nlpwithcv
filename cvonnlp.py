from screen import get_detected_objects
from bert import wrap_text_dynamic, answer_question, extract_keywords
from gpt import correct_grammar


def initial_context():
    initial = get_detected_objects()
    initial_context_text = "Following objects are on the screen: " + initial
    return str(initial_context_text)
#print(initial_context())
def prominent(text):
    keywords = extract_keywords(text)
    return str(keywords)


def aboutobj(text):
    prominentobj = prominent(text)
    return str(wrap_text_dynamic(prominentobj))

def complete_context():
    initial_text = initial_context()
    about_text = aboutobj(initial_text)
    totalcomp = initial_text + '\n\n' + "details :"+about_text
    return str(totalcomp)
#print(complete_context())

"""def Wosc(question):
    ques = question
    abstract = complete_context()
    answer = answer_question(ques, abstract)
    correct_answer = correct_grammar(answer)
    return answer
"""


def Wosc(inp):
    if inp == "yes":pass
    abstract = complete_context()
    print("where is everything i Know about whats on screen")
    print (abstract)
    im_run = True
    while im_run:
        inp = str(input("do you want to continue on this topic or go to next topic ?"))
        if str(inp) == "next topic":
            im_run = False
        else:
            ques = str(input("whats your question"))
            answer = answer_question(ques, abstract)
            correct_answer = correct_grammar(answer)
            print(correct_answer)

    return None


"""i = str(input("shall we start?"))
Wosc(i)"""