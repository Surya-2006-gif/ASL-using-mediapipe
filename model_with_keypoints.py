import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import os
import random
from keras import layers, models
from keras import regularizers
import matplotlib.pyplot as plt

# Data loading and preprocessing
columns = ['label', 'thumb_tip_x', 'thumb_tip_y', 'index_tip_x', 'index_tip_y',
           'middle_tip_x', 'middle_tip_y', 'ring_tip_x', 'ring_tip_y',
           'pinky_tip_x', 'pinky_tip_y']

a = pd.read_csv('keypoints/A.csv', names=columns)
b = pd.read_csv('keypoints/B.csv', names=columns)
c = pd.read_csv('keypoints/C.csv', names=columns)
df = pd.concat([a, b, c], ignore_index=True)

x_keypoints = df.drop(columns=['label']).values
y_labels_from_csv = df['label'].values 

label_dict = {'A': 0, 'B': 1, 'C': 2}

folder = "data_new"
images = []
img_labels = []
sub_folders = ['A', "B", "C"]
for i in sub_folders:
    total_path = os.path.join(folder, i)
    label = label_dict[i]
    for img in os.listdir(total_path):
        img_path = os.path.join(total_path, img)
        images.append(img_path)
        img_labels.append(label)

random.seed(42)
combined = list(zip(images, img_labels, x_keypoints))
random.shuffle(combined)
images, labels, x_keypoints = zip(*combined)

images = list(images)
labels = np.array(labels)
x_keypoints = np.array(x_keypoints, dtype=np.float32)

# Dataset creation
def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [300, 300])  # Match model input size
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

# Create named inputs for better model compatibility
def create_combined_dataset():
    img_ds = tf.data.Dataset.from_tensor_slices((images, labels))
    img_ds = img_ds.map(load_and_preprocess)
    
    kp_ds = tf.data.Dataset.from_tensor_slices((x_keypoints, labels))
    
    # Combine with proper naming
    def combine_data(img_data, kp_data):
        return {'image_input': img_data[0], 
                'keypoints_input': kp_data[0]}, img_data[1]
    
    return tf.data.Dataset.zip((img_ds, kp_ds)).map(combine_data)

ds_combined = create_combined_dataset()

# Train/val split
total_samples = len(images)
val_size = int(0.2 * total_samples)
train_ds = ds_combined.skip(val_size)
val_ds = ds_combined.take(val_size)

def pipeline(ds):
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(128)
    ds = ds.cache()
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

train_ds = pipeline(train_ds)
val_ds = pipeline(val_ds)

# Model definition with named inputs
def multimodal_model():
    # Inputs
    input_1 = layers.Input(shape=(300, 300, 3), name='image_input')
    input_2 = layers.Input(shape=(10,), name='keypoints_input')
    
    # Image branch
    x = layers.Conv2D(11, 3, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_1)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(21, 3, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='relu', kernel_regularizer=regularizers.l1(0.01))(x)
    
    # Keypoint branch
    y = layers.Dense(10, activation='relu', kernel_regularizer=regularizers.l1(0.01))(input_2)

    # Attention fusion mechanism
    alpha_img = layers.Dense(10, activation='sigmoid')(x)
    alpha_kp  = layers.Dense(10, activation='sigmoid')(y)

    sum_alpha = layers.Add()([alpha_img, alpha_kp])
    
    alpha_img_norm = layers.Lambda(lambda a: a[0] / (a[1] + 1e-6))([alpha_img, sum_alpha])
    alpha_kp_norm  = layers.Lambda(lambda a: a[0] / (a[1] + 1e-6))([alpha_kp, sum_alpha])

    x_att = layers.Multiply()([alpha_img_norm, x])
    y_att = layers.Multiply()([alpha_kp_norm, y])

    fused = layers.Add()([x_att, y_att])

    # Output
    output = layers.Dense(3, activation='softmax')(fused)

    model = models.Model(inputs=[input_1, input_2], outputs=output)
    return model

model = multimodal_model()
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save("multimodal_model.keras")
print("Model saved as multimodal_model.keras")