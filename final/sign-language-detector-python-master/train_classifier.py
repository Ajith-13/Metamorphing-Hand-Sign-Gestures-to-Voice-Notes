import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('A:/finalpro/final/data.pickle', 'rb'))

max_length = max(len(seq) for seq in data_dict['data'])

data_padded = []
for seq in data_dict['data']:
    pad_length = max_length - len(seq)
    padded_seq = np.pad(seq, (0, pad_length), mode='constant')
    data_padded.append(padded_seq)

data = np.asarray(data_padded)
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, 
test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

with open('A:/finalpro/final/model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

