import tensorflow as tf
import keras
from keras import layers, models
import os
from matplotlib import pyplot as plt
import numpy as np
# Define paths (you've already done this part)
a_data_path = os.path.join("data", "A")
b_data_path = os.path.join("data", "B")
c_data_path = os.path.join("data", "C")

# Build the CNN model
inputs = layers.Input(shape=(300, 300, 3))
x = layers.Conv2D(filters=11, kernel_size=3, activation='relu',kernel_regularizer=keras.regularizers.l2(0.005))(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=21, kernel_size=3, activation='relu',kernel_regularizer=keras.regularizers.l2(0.005))(x)
x = layers.Flatten()(x)
outputs = layers.Dense(4, activation='softmax')(x)

model = models.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Load datasets (training + validation)
train_dataset = tf.keras.utils.image_dataset_from_directory(
    directory="data",
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(300, 300),  # Must match input shape
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset="training"
)

print(train_dataset.class_names)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    directory="data",
    labels="inferred",
    label_mode="int",
    batch_size=128,
    image_size=(300, 300),  # Must match input shape
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset="validation"
)

# (Optional) Improve performance by prefetching data
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs= 5
)

plt.plot(np.arange(5),history.history['loss'], label='accuracy',color='blue')
plt.plot(np.arange(5),history.history['val_loss'], label='val_accuracy',color='orange')
plt.show()

model.save("model.keras")
print("Model saved as model.keras")