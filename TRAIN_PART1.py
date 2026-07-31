import numpy as np
import pandas as pd
import pickle
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.feature_extraction.text import CountVectorizer

IELTS_SCORES = np.array([5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0])

df = pd.read_csv(r"D:\TRAIN_MODEL_SPEAKING\SPEAKING_DATASET.csv")
ds = df[df['CONTEXT_CLUSTER'] == 1].copy()
def prepare_data(data=ds):
    group = ds.groupby('ID_CLUSTER').agg({
        'QUESTION': lambda x:" ".join(x.astype(str)),
        'ANSWER': lambda x:" ".join(x.astype(str)),
        'CONTEXT': 'first',
        'SCORE': 'first'
    }).reset_index()
    question = group['QUESTION'].astype(str).str.lower().values
    answer = group['ANSWER'].astype(str).str.lower().values
    context = group['CONTEXT'].astype(str).str.lower().values
    
    raw_scores = group['SCORE'].astype(float).values
    y = np.array([np.where(IELTS_SCORES == s)[0][0] for s in raw_scores])

    vectorizer = CountVectorizer(stop_words='english')
    vectorizer.fit(np.concatenate([question,answer,context]))

    q_vec = vectorizer.transform(question).toarray()
    a_vec = vectorizer.transform(answer).toarray()
    c_vec = vectorizer.transform(context).toarray()

    X = np.concatenate([q_vec,a_vec,c_vec],axis=1)
    with open(r'D:\TRAIN_MODEL_SPEAKING\vectorizer_part1.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    history_part1 = {}
    for _,row in group.iterrows():
        id_cluster = row['ID_CLUSTER']
        if id_cluster not in history_part1:
            history_part1[id_cluster] = {
                "question":[row['QUESTION']],
                "answer":[row['ANSWER']],
                "context": ''
            }
    with open(r'D:\TRAIN_MODEL_SPEAKING\history_part1.pkl', 'wb') as p:
        pickle.dump(history_part1, p)
    return X,y

X,y = prepare_data()
model = keras.Sequential([layers.Input(shape=(X.shape[1],)),
                          layers.Dense(9,activation='softmax')])
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X,y,epochs=1000,verbose=1)

model.save(r'D:\TRAIN_MODEL_SPEAKING\model_part1.keras')
print('Đã lưu mô hình')
