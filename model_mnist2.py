import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# MNIST-Daten laden und vorbereiten
(x_train, y_train), _ = mnist.load_data()
x_train = x_train / 255.0
y_train = to_categorical(y_train, 10)

# Modell definieren
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])

# Modell kompilieren und trainieren
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# Modell korrekt speichern
model.save('my_model2.keras')
print("Modell wurde erfolgreich gespeichert.")