import re

import pandas
import numpy


def set_title_column(train, test): 
    def _get_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        if title_search:
            return title_search.group(1)
        return ""

    def _set_titles(df):
        titles = df["Name"].apply(_get_title)
        for k,v in title_mapping.items():
            titles[titles == k] = v
        df["Title"] = titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 3}
    _set_titles(train)
    _set_titles(test)
    for col_i in set(title_mapping.values()):
        train.insert(len(train.columns), "Title_" + str(col_i), [1 if x == col_i else 0 for x in train['Title']])
        test.insert(len(test.columns), "Title_" + str(col_i), [1 if x == col_i else 0 for x in test['Title']])
    train.drop(['Title'], 1)
    test.drop(['Title'], 1)


def maxminscale(col_data, max_val, min_val):
    return (col_data - min_val) / (max_val - min_val)

def normalise(train, test, columns):
    for c in columns:
#         norm = MinMaxScaler()
#         norm.fit(numpy.concatenate([train[c], test[c]]))
        max_val, min_val = numpy.concatenate([train[c], test[c]]).max(), numpy.concatenate([train[c], test[c]]).min()
        train[c] = maxminscale(train[c], max_val, min_val)
        test[c] = maxminscale(test[c], max_val, min_val)


def replace_nan_age(data):
    #data['Age'].fillnull(data.Age.mean())
    data.loc[data.Age.isnull(), 'Age'] = data.Age.mean()


def replace_nan_fair(data):
    #data['Fare'].fillnull(data.Fare.median())
    data.loc[data.Fare.isnull(), 'Fare'] = data.Fare.median()


def make_dummies(data, columns):
    #return pandas.get_dummies(pandas.get_dummies(data, columns = ['Pclass', 'SibSp', 'Parch']))
    return pandas.get_dummies(data, columns = columns)


def drop_columns(data, to_drop):
    return data.drop(to_drop, 1)


def set_family_size(data):
    data["FamilySize"] = data["SibSp"] + data["Parch"]

    
def stacking_model_predict(clf, train_X, train_y, test):
    clf.fit(train_X, train_y)
    return clf.predict_proba(test)


def insert_predictions(df, pred, colname):
    for col_i in range(pred.shape[1]):
        df.insert(len(df.columns), colname + str(col_i), pred[:,col_i])
