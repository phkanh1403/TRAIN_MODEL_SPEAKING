
import speech_recognition as sr
import pyttsx3
import time
import random
import numpy as np
import pandas as pd
import pickle
from tensorflow import keras

df = pd.read_csv(r"D:\TRAIN_MODEL_SPEAKING\SPEAKING_DATASET.csv")

class IELTSCHATBOT:
    def __init__(self):
        self.model_config = {
            "PART1": {
                'model_path': 'model_part1.keras',
                'vectorizer_path': 'vectorizer_part1.pkl',
                'history_path': 'history_part1.pkl'
            },
            "PART2": {
                'model_path': 'model_part2.keras',
                'vectorizer_path': 'vectorizer_part2.pkl',
                'history_path': 'history_part2.pkl'
            },
            "PART3": {
                'model_path': 'model_part3.keras',
                'vectorizer_path': 'vectorizer_part3.pkl',
                'history_path': 'history_part3.pkl'
            }
        }

        self.model=None
        self.vectorizer=None
        self.history=None

    def load_model(self, part):
        config = self.model_config.get(part)
        if config:
            self.model = keras.models.load_model(config['model_path'])
            with open(config['vectorizer_path'],'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(config['history_path'],'rb') as f:
                self.history = pickle.load(f)
        else:
            print("Invalid part specified. Choose from PART1, PART2, or PART3.")

    def get_response(self, user_input):
        question = str(user_input['question']).lower()
        answer = str(user_input['answer']).lower()
        context = str(user_input['context']).lower()

        q_vec = self.vectorizer.transform([question]).toarray()
        a_vec = self.vectorizer.transform([answer]).toarray()
        c_vec = self.vectorizer.transform([context]).toarray()
        X = np.concatenate([q_vec, a_vec, c_vec], axis=1)

        IELTS_SCORES_PREDICT_RAW = self.model.predict(X)
        IELTS_SCORES_PREDICT_ID = np.argmax(IELTS_SCORES_PREDICT_RAW, axis=1)
        IELTS_SCORES_CLASS = np.array([5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0])
        IELTS_SCORES_PREDICT = IELTS_SCORES_CLASS[IELTS_SCORES_PREDICT_ID]

        return IELTS_SCORES_PREDICT[0]

class IELTSMAIN:
    def __init__(self):
        self.chatbot = IELTSCHATBOT()

        self.tts = pyttsx3.init()
        voices = self.tts.getProperty('voices')
        self.tts.setProperty('rate',165)
        self.recognizer = sr.Recognizer()
        self.scores = []

    def speak(self, text):
        self.tts.say(text)
        self.tts.runAndWait()

    def listen(self):
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source,duration=0.8)
            self.recognizer.pause_threshold = 10
            try:
                audio = self.recognizer.listen(source,timeout=45,phrase_time_limit=360)
                text = self.recognizer.recognize_google(audio,language='en-US')
                print(f"You said: {text}")
                return text
            except Exception:
                print("Sorry, I could not understand. Please try again.")
                return None

    def run(self):
        part = input('Enter the part of the IELTS exam (PART1, PART2, PART3): ').strip().upper()
        self.chatbot.load_model(part)
        topic = df[df['CONTEXT_CLUSTER']==int(part[4])]['CONTEXT'].unique()
        selected_topic = random.choice(topic)
        print(f"Selected topic: {selected_topic} and {part}")

        question = df[(df['CONTEXT']==selected_topic) & (df['CONTEXT_CLUSTER']==int(part[4]))]['QUESTION'].unique()
        random.shuffle(question)
        if part == 'PART1':
            selected_questions = question[:4]
        elif part == 'PART2':
            selected_questions = question[:1]
        elif part == 'PART3':
            selected_questions = question[:3]

        for current_question in selected_questions:
            print(f"Question: {current_question}")
            self.speak(current_question)
            user_input = self.listen()
            score_input = {
                'question': current_question,
                'answer': user_input,
                'context':selected_topic
            }
            score = self.chatbot.get_response(score_input)
            self.scores.append(score)

        if self.scores:
            average_score = round(np.mean(self.scores),1)
            print(f"Your average IELTS score is: {average_score}")


app = IELTSMAIN()
app.run()

        