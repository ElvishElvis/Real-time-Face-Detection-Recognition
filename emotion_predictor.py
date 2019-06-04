import pandas as pd
from sklearn.linear_model import LogisticRegression


train = pd.read_csv('emotions.csv', engine='python', error_bad_lines=False,
                    warn_bad_lines=False, header=None)
train = train[train[136].isna() == False]
diction = {'neutral': 1, 'surprise': 2, 'disgust': 3, 'anger': 4, 'happy': 5}
train[136] = train[136].replace(diction)
train = train.apply(lambda x: x.astype(int), axis=1)
x = train[[i for i in range(136) if i % 2 == 0]]
y = train[[i for i in range(137) if i % 2 == 1]]
try:
    x = x.subtract(x.apply(lambda i: min(i), axis=1), axis=0)
    y = y.subtract(y.apply(lambda i: min(i), axis=1), axis=0)
    train = pd.concat([x, y, train[136]], axis=1).T.sort_index().T
    train = train.apply(lambda x: x.apply(lambda y: int(y)), axis=1)
    X_train = train.drop(136, axis=1)
    y_train = train[136]
    mult = LogisticRegression()
    mult.fit(X_train, y_train)
    back = {1: 'neutral', 2: 'surprise', 3: 'disappoint', 4: 'anger', 5: 'happy'}
except KeyError:
    pass

def output(row):
    try:
        row = pd.DataFrame(row).T.astype(int)
        x_row = row[[i for i in range(136) if i % 2 == 0]]
        y_row = row[[i for i in range(136) if i % 2 == 1]]

        x_row = x_row.subtract(x_row.apply(lambda i: min(i), axis=1), axis=0)
        y_row = y_row.subtract(y_row.apply(lambda i: min(i), axis=1), axis=0)
        row = pd.concat([x_row, y_row], axis=1).T.sort_index().T
        return pd.DataFrame(mult.predict(row)).replace(back).loc[0][0]
    except KeyError:
        pass
