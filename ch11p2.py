#Om Gole, Period 6, Gabor
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score

# Load iris dataset
data = load_iris()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to create the model
def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=4, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    return model
epochs = 15
# Create the KerasClassifier wrapper & Change epochs to test accuracy (5 - .7/ 15 - .933)
model = KerasClassifier(build_fn=create_model, epochs=15, batch_size=16, verbose=0)

# Train the model
model.fit(X_train, y_train)

# Monte Carlo Dropout
T = 10
predictions = np.zeros((T, X_test.shape[0], 3))
for t in range(T):
    predictions[t] = model.predict_proba(X_test)

prediction_probabilities = predictions.mean(axis=0)
prediction_classes = prediction_probabilities.argmax(axis=-1)

# Calculate accuracy
accuracy = accuracy_score(y_test, prediction_classes)
print(f"Accuracy: {accuracy}", f"This was conducted with an {epochs} epochs")