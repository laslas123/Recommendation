import numpy as np
from tensorflow import keras
import os

# Set the dataset path and category labels
data_dir = 'train'
class_names = ['Baby Onesie', 'Coat', 'Dress', 'Dungarees', 'Hoodie', 'Jeans', 'Pants', 'Shirt',
                 'Shorts', 'Skirt', 'Strap dress', 'Suit set', 'Sweater', 'T-shirt', 'Vest',
                 'Wedding Dress']

num_classes = len(class_names)

# load the dataset
images = []
labels = []

for class_index, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        try:
            image = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
            image = keras.preprocessing.image.img_to_array(image)
            images.append(image)
            labels.append(class_index)
        except Exception as e:
            print(f"Error loading image: {image_path}\n{e}")

images = np.array(images)
labels = np.array(labels)

# Normalized
images = images / 255.0

# Build the model
model = keras.Sequential([
    keras.layers.Input(shape=(224, 224, 3)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(images, labels, epochs=10)
# Save model weights
model.save_weights('model_weights.h5')

