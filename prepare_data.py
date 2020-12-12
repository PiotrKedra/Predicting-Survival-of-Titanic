import pandas as pd
import random
import re


def prepare_data(data):
    clean(data)
    map_to_numbers(data)


def clean(data):
    """ PassengerId and Ticket number are just ids so we drop them"""
    del data['PassengerId']
    del data['Ticket']
    """
    SibSp -> number of siblings/spouses aboard, Parch -> number of children/parents aboard
    from these 2 columns we make one that tell give us number of relatives aboard + 1 (specific person)
    """
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    del data['SibSp']
    del data['Parch']

    """ we change nulls for S (Southampton), it takes around 70% of all values"""
    data['Embarked'].fillna('S', inplace=True)

    """ fill nulls with random number between (average - standard_deviation) and (average + standard_deviation)"""
    fill_age_nulls(data)

    """ fill nulls with median"""
    fare_median = data['Fare'].median()
    data['Fare'].fillna(fare_median, inplace=True)

    """ 
    lots of nulls here
    we change each null for 0, and not nulls we map to just deck numbers A,B,..,G and then to numbers 1..7
    """
    data['Cabin'] = data['Cabin'].apply(lambda x: change_cabin_number_to_deck_number(x))

    """
    from name we extract just title e.g. Miss. Mr. Rev. 
    we delete Name and create Title column
    """
    data['Title'] = data['Name'].apply(lambda x: re.search(r'\S*\.', x)[0])
    del data['Name']


def fill_age_nulls(data):
    age_mean = data['Age'].mean()
    age_std = data['Age'].std()
    age_min = age_mean - age_std
    age_max = age_mean + age_std
    data['Age'].fillna(float(int(random.uniform(age_min, age_max))), inplace=True)


def change_cabin_number_to_deck_number(number_str):
    deck = {"A": 8, "B": 7, "C": 6, "D": 5, "E": 4, "F": 3, "G": 2, "T": 1}
    if pd.isna(number_str):
        return 0
    return deck[number_str[0]]


def map_to_numbers(data):
    """male = 0, female = 1"""
    data['Sex'] = data['Sex'].apply(lambda x: 0 if x == 'male' else 1)

    all_titles = {'Mr.': 1, 'Mrs.': 2, 'Miss.': 3, 'Master.': 4, 'Don.': 5, 'Rev.': 6,
                  'Dr.': 7, 'Mme.': 8, 'Ms.': 9, 'Major.': 10, 'Lady.': 11, 'Sir.': 12,
                  'Mlle.': 13, 'Col.': 14, 'Capt.': 15, 'Countess.': 16, 'Jonkheer.': 17, 'Dona.': 18}
    data['Title'] = data['Title'].apply(lambda x: all_titles[x])

    all_embarked = {'S': 0, 'Q': 1, 'C': 2}
    data['Embarked'] = data['Embarked'].apply(lambda x: all_embarked[x])

    group_fare(data)

    group_age(data)


def group_fare(data):
    data.loc[data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare'] = 2
    data.loc[(data['Fare'] > 31) & (data['Fare'] <= 99), 'Fare'] = 3
    data.loc[(data['Fare'] > 99) & (data['Fare'] <= 170), 'Fare'] = 4
    data.loc[(data['Fare'] > 170) & (data['Fare'] <= 270), 'Fare'] = 5
    data.loc[data['Fare'] > 270, 'Fare'] = 6
    data['Fare'] = data['Fare'].astype(int)


def group_age(data):
    data['Age'] = data['Age'].astype(int)
    data.loc[data['Age'] <= 4, 'Age'] = 0
    data.loc[(data['Age'] > 4) & (data['Age'] <= 11), 'Age'] = 1
    data.loc[(data['Age'] > 11) & (data['Age'] <= 18), 'Age'] = 2
    data.loc[(data['Age'] > 18) & (data['Age'] <= 22), 'Age'] = 3
    data.loc[(data['Age'] > 22) & (data['Age'] <= 27), 'Age'] = 4
    data.loc[(data['Age'] > 27) & (data['Age'] <= 33), 'Age'] = 5
    data.loc[(data['Age'] > 33) & (data['Age'] <= 40), 'Age'] = 6
    data.loc[(data['Age'] > 40) & (data['Age'] <= 50), 'Age'] = 7
    data.loc[(data['Age'] > 50) & (data['Age'] <= 60), 'Age'] = 8
    data.loc[data['Age'] > 60, 'Age'] = 9

