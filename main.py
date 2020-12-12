import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU

from prepare_data import prepare_data

train = pd.read_csv('titanic/train.csv')
test = pd.read_csv('titanic/test.csv')

prepare_data(train)
prepare_data(test)

X_train = train.drop('Survived', axis=1)
Y_train = train['Survived']
X_test = test.copy()

# Logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

# Artificial Neural Networks
model = Sequential()
model.add(Dense(input_dim=8, units=4))
model.add(LeakyReLU())
model.add(Dense(units=1))
model.add(LeakyReLU())
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train, validation_split=0.20, batch_size=10, epochs=500, verbose=0, shuffle=True)

acc_ann = round(history.history['accuracy'][-1] * 100, 2)

print('Models accuracy: ')
print('logistic regression: ' + str(acc_log))
print('random forest: ' + str(acc_random_forest))
print('neural networks: ' + str(acc_ann))
print()

# Pclass  Sex  Age  Fare  Deck  Embarked  FamilySize  Title
# jacki 3rd male 20 0$ G S 1 Mr
# rose 1st female 17 200$ B S 3 Miss
jacki_and_rose = [[3, 0, 3, 0, 7, 0, 1, 1],
                  [1, 1, 2, 5, 2, 0, 3, 3]]
predictions = model.predict(jacki_and_rose)
jack = round(predictions.flat[0] * 100, 2)
rose = round(predictions.flat[1] * 100, 2)
print('Chance of survival')
print('Jack\'s: ' + str(jack))
print('Rose\'s: ' + str(rose))


