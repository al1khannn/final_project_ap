# Recommending Films Based on the Facial Expression of a Person

This project is an end-to-end application designed to analyze emotions from images and recommend movies based on the detected emotions. It leverages TensorFlow for building a convolutional neural network (CNN) to classify emotions from facial expressions in images. The project utilizes the FER2013 dataset for training the emotion recognition model.  features a movie recommendation system that suggests movies based on the detected emotion.

## Getting Started

### Prerequisites

- Google Colab
- Google Drive with enough space for dataset and trained model

### Setup Instructions

1. **Mount Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')
```

2. **Navigate to the Project Directory**

```python
%cd drive/MyDrive/final_proj_dataset1
```

3. **Unarchive the Dataset**

```python
import tarfile

fname = 'fer2013.tar.gz'
if fname.endswith("tar.gz"):
    tar = tarfile.open(fname, "r:gz")
    tar.extractall()
    tar.close()
elif fname.endswith("tar"):
    tar = tarfile.open(fname, "r:")
    tar.extractall()
    tar.close()
```

4. **Install Required Libraries**

Ensure all necessary Python libraries are installed.

```python
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot
```

### Data Preprocessing

The data preprocessing stage is crucial for preparing the raw FER2013 dataset for the model training process. This stage involves several steps to ensure the data is in the right format and structure for the convolutional neural network (CNN) to process effectively. Below is a detailed walkthrough of the data preprocessing steps.

#### Load and Explore the FER2013 Dataset

1. **Load the Dataset:**

   Begin by loading the FER2013 dataset into a Pandas DataFrame. This dataset contains grayscale images of facial expressions categorized into seven emotions: anger, disgust, fear, happiness, sadness, surprise, and neutral.

   ```python
   import pandas as pd

   df = pd.read_csv('fer2013/fer2013.csv')
   df.head()
   ```

2. **Understand the Data Structure:**

   The dataset consists of three columns: `emotion`, which is a numeric label representing the emotion category; `pixels`, which contains the pixel values for each image in a string format; and `Usage`, which indicates the data split (training, public test, private test).

   Explore the dataset to understand the distribution of different emotions and the overall composition of the dataset. This exploration helps in identifying any imbalances in the dataset that might require addressing.

#### Preprocess Images for Training

1. **Pixel String to Array Conversion:**

   Convert the pixel strings in the `pixels` column into arrays. Each image is represented as a 48x48 pixel grayscale image, so you need to reshape the string into a 48x48 numpy array. Additionally, these pixel values should be converted from strings to floats for computation.

   ```python
   import numpy as np

   img_array = df.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))
   ```

2. **Normalize the Pixel Values:**

   Normalize the pixel values to the range 0 to 1 to aid in the neural network's convergence. Pixel values are originally in the range 0-255, so dividing by 255 achieves this normalization.

   ```python
   img_array = np.stack(img_array, axis=0) / 255.0
   ```

3. **Prepare Labels:**

   Extract the emotion labels from the dataset. These labels are already in a numeric format suitable for classification.

   ```python
   labels = df.emotion.values
   ```

4. **Train-Test Split:**

   Split the dataset into training and testing sets to evaluate the model's performance on unseen data. It's common to use a split ratio like 90% for training and 10% for testing.

   ```python
   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(img_array, labels, test_size=0.1)
   ```

### Model Building: Design and Compile a CNN Model for Emotion Recognition

Building an effective model for emotion recognition involves designing a convolutional neural network (CNN) architecture that can accurately classify different emotions based on facial expressions in images. This section details the process of designing and compiling a CNN model tailored for emotion recognition from the FER2013 dataset.

#### Designing the CNN Architecture

1. **Input Layer:**

   The input layer should match the shape of the preprocessed images. Since the FER2013 dataset consists of 48x48 pixel grayscale images, the input shape will be `(48, 48, 1)`, where 1 indicates a single color channel (grayscale).

2. **Convolutional Layers:**

   Convolutional layers are the core building blocks of a CNN. They are responsible for extracting features from the images through the use of various filters. Start with convolutional layers that have a small number of filters (e.g., 32) and gradually increase this number in subsequent layers to allow the network to capture more complex features. Use a kernel size of `(3, 3)` for these layers.

3. **Activation Function:**

   Use the ReLU (Rectified Linear Unit) activation function for the convolutional layers. ReLU helps in adding non-linearity to the model, enabling it to learn more complex patterns.

4. **Pooling Layers:**

   Max pooling layers are used to reduce the spatial dimensions of the output from the previous convolutional layers. They help in reducing the number of parameters and computation in the network, and also in controlling overfitting. A common choice is a `(2, 2)` pool size for these layers.

5. **Flattening Layer:**

   Before connecting to the fully connected layers, flatten the output from the convolutional layers. This converts the 2D feature maps into a 1D feature vector.

6. **Fully Connected (Dense) Layers:**

   Dense layers further process the features extracted by the convolutional layers. Include a large dense layer with, for example, 1000 units, followed by a ReLU activation function. This is followed by the output layer.

7. **Output Layer:**

   The output layer should have as many neurons as the number of emotion categories to be classified (e.g., 7 for the FER2013 dataset). Use the softmax activation function for this layer to obtain the probability distribution over the classes.

#### Model Architecture Example

```python
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same'),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])
```

#### Compiling the Model

Compile the model with an appropriate optimizer, loss function, and metrics for classification. A common choice for the optimizer is RMSprop or Adam, with a learning rate of around `0.0001`. Since this is a multi-class classification problem, use the `sparse_categorical_crossentropy` loss function. For metrics, accuracy is a straightforward choice to evaluate the model's performance.

```python
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

After designing and compiling your CNN model, it is ready to be trained on the preprocessed dataset. This CNN architecture incorporates convolutional layers for feature extraction, pooling layers for dimensionality reduction, and dense layers for classification, making it well-suited for emotion recognition tasks.

### Model Building, Training, Evaluation, and Movie Recommendations

This comprehensive section walks through the steps of building, training, evaluating a CNN model for emotion recognition, and implementing a movie recommendation system based on detected emotions.

#### Model Building: Design and Compile a CNN Model for Emotion Recognition

**Designing the CNN Architecture:**

1. **Input Layer:** Shape `(48, 48, 1)` for grayscale images from the FER2013 dataset.
2. **Convolutional and Pooling Layers:** Use multiple convolutional layers with increasing numbers of filters (e.g., 32, 64, 128), ReLU activation, followed by max pooling layers to extract and downsample features.
3. **Flattening Layer:** Flatten the output for dense layer processing.
4. **Dense Layers:** A large dense layer (e.g., 1000 units) with ReLU activation followed by the output layer.
5. **Output Layer:** 7 units (for 7 emotions) with softmax activation.

**Compilation:**

- **Optimizer:** RMSprop or Adam with a learning rate of `0.0001`.
- **Loss Function:** `sparse_categorical_crossentropy`.
- **Metrics:** `accuracy`.

```python
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

#### Model Training: Implement Model Checkpoints and Callbacks

**Model Checkpoints and Callbacks:**

1. **ModelCheckpoint:** Saves the best model based on validation accuracy.
2. **EarlyStopping:** Stops training when the validation loss stops improving, preventing overfitting.

```python
import os
import tensorflow as tf

checkpoint_path = "model_checkpoint/best_model.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_best_only=True,
                                                 monitor='val_accuracy',
                                                 verbose=1)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

callbacks_list = [cp_callback, early_stop]
```

**Training:**

- Use the processed dataset for training.
- Include validation split for monitoring overfitting.
- Apply callbacks for checkpointing and early stopping.

Output: 
Epoch 1/50
907/909 [============================>.] - ETA: 0s - loss: 1.6049 - accuracy: 0.3799
Epoch 1: val_accuracy did not improve from 0.66393
909/909 [==============================] - 10s 11ms/step - loss: 1.6049 - accuracy: 0.3799 - val_loss: 1.4965 - val_accuracy: 0.4258
Epoch 2/50
907/909 [============================>.] - ETA: 0s - loss: 1.4476 - accuracy: 0.4469
Epoch 2: val_accuracy did not improve from 0.66393
909/909 [==============================] - 10s 11ms/step - loss: 1.4472 - accuracy: 0.4473 - val_loss: 1.3833 - val_accuracy: 0.4754
Epoch 3/50
907/909 [============================>.] - ETA: 0s - loss: 1.3446 - accuracy: 0.4931
Epoch 3: val_accuracy did not improve from 0.66393
909/909 [==============================] - 10s 11ms/step - loss: 1.3444 - accuracy: 0.4932 - val_loss: 1.3506 - val_accuracy: 0.4912
Epoch 4/50
904/909 [============================>.] - ETA: 0s - loss: 1.2494 - accuracy: 0.5271
Epoch 4: val_accuracy did not improve from 0.66393
909/909 [==============================] - 11s 12ms/step - loss: 1.2492 - accuracy: 0.5270 - val_loss: 1.2645 - val_accuracy: 0.5226
Epoch 5/50
909/909 [==============================] - ETA: 0s - loss: 1.1550 - accuracy: 0.5674
Epoch 5: val_accuracy did not improve from 0.66393
909/909 [==============================] - 11s 12ms/step - loss: 1.1550 - accuracy: 0.5674 - val_loss: 1.1580 - val_accuracy: 0.5652
Epoch 6/50
909/909 [==============================] - ETA: 0s - loss: 1.0553 - accuracy: 0.6083
Epoch 6: val_accuracy did not improve from 0.66393
909/909 [==============================] - 9s 10ms/step - loss: 1.0553 - accuracy: 0.6083 - val_loss: 0.9991 - val_accuracy: 0.6334
Epoch 7/50
903/909 [============================>.] - ETA: 0s - loss: 0.9518 - accuracy: 0.6524
Epoch 7: val_accuracy did not improve from 0.66393
909/909 [==============================] - 9s 10ms/step - loss: 0.9515 - accuracy: 0.6523 - val_loss: 0.9422 - val_accuracy: 0.6604
Epoch 8/50
907/909 [============================>.] - ETA: 0s - loss: 0.8302 - accuracy: 0.7026
Epoch 8: val_accuracy improved from 0.66393 to 0.73831, saving model to checkpoint/best_model.h5
/usr/local/lib/python3.10/dist-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(
909/909 [==============================] - 11s 12ms/step - loss: 0.8304 - accuracy: 0.7027 - val_loss: 0.7898 - val_accuracy: 0.7383
Epoch 9/50
908/909 [============================>.] - ETA: 0s - loss: 0.6959 - accuracy: 0.7533
Epoch 9: val_accuracy improved from 0.73831 to 0.74351, saving model to checkpoint/best_model.h5
909/909 [==============================] - 11s 12ms/step - loss: 0.6959 - accuracy: 0.7533 - val_loss: 0.7452 - val_accuracy: 0.7435
Epoch 10/50
905/909 [============================>.] - ETA: 0s - loss: 0.5524 - accuracy: 0.8081
Epoch 10: val_accuracy improved from 0.74351 to 0.80742, saving model to checkpoint/best_model.h5
909/909 [==============================] - 10s 12ms/step - loss: 0.5526 - accuracy: 0.8080 - val_loss: 0.6167 - val_accuracy: 0.8074
Epoch 11/50
907/909 [============================>.] - ETA: 0s - loss: 0.4157 - accuracy: 0.8563
Epoch 11: val_accuracy improved from 0.80742 to 0.83826, saving model to checkpoint/best_model.h5
909/909 [==============================] - 9s 10ms/step - loss: 0.4159 - accuracy: 0.8562 - val_loss: 0.5622 - val_accuracy: 0.8383
Epoch 12/50
904/909 [============================>.] - ETA: 0s - loss: 0.3014 - accuracy: 0.8987
Epoch 12: val_accuracy improved from 0.83826 to 0.85708, saving model to checkpoint/best_model.h5
909/909 [==============================] - 10s 11ms/step - loss: 0.3017 - accuracy: 0.8986 - val_loss: 0.5286 - val_accuracy: 0.8571
Epoch 13/50
908/909 [============================>.] - ETA: 0s - loss: 0.2184 - accuracy: 0.9280
Epoch 13: val_accuracy improved from 0.85708 to 0.87931, saving model to checkpoint/best_model.h5
909/909 [==============================] - 9s 10ms/step - loss: 0.2184 - accuracy: 0.9280 - val_loss: 0.5047 - val_accuracy: 0.8793
Epoch 14/50
909/909 [==============================] - ETA: 0s - loss: 0.1613 - accuracy: 0.9496
Epoch 14: val_accuracy improved from 0.87931 to 0.88340, saving model to checkpoint/best_model.h5
909/909 [==============================] - 10s 11ms/step - loss: 0.1613 - accuracy: 0.9496 - val_loss: 0.6004 - val_accuracy: 0.8834
Epoch 15/50
908/909 [============================>.] - ETA: 0s - loss: 0.1257 - accuracy: 0.9614
Epoch 15: val_accuracy improved from 0.88340 to 0.88600, saving model to checkpoint/best_model.h5
909/909 [==============================] - 10s 11ms/step - loss: 0.1257 - accuracy: 0.9614 - val_loss: 0.6208 - val_accuracy: 0.8860
Epoch 16/50
907/909 [============================>.] - ETA: 0s - loss: 0.1044 - accuracy: 0.9690
Epoch 16: val_accuracy improved from 0.88600 to 0.89634, saving model to checkpoint/best_model.h5
909/909 [==============================] - 10s 11ms/step - loss: 0.1043 - accuracy: 0.9690 - val_loss: 0.6105 - val_accuracy: 0.8963
Epoch 17/50
905/909 [============================>.] - ETA: 0s - loss: 0.0882 - accuracy: 0.9747
Epoch 17: val_accuracy improved from 0.89634 to 0.89646, saving model to checkpoint/best_model.h5
909/909 [==============================] - 11s 12ms/step - loss: 0.0883 - accuracy: 0.9746 - val_loss: 0.6185 - val_accuracy: 0.8965
Epoch 18/50
905/909 [============================>.] - ETA: 0s - loss: 0.0765 - accuracy: 0.9781
Epoch 18: val_accuracy improved from 0.89646 to 0.89727, saving model to checkpoint/best_model.h5
909/909 [==============================] - 11s 12ms/step - loss: 0.0769 - accuracy: 0.9779 - val_loss: 0.6440 - val_accuracy: 0.8973
Epoch 19/50
904/909 [============================>.] - ETA: 0s - loss: 0.0756 - accuracy: 0.9795
Epoch 19: val_accuracy improved from 0.89727 to 0.90105, saving model to checkpoint/best_model.h5
909/909 [==============================] - 10s 11ms/step - loss: 0.0758 - accuracy: 0.9794 - val_loss: 0.6170 - val_accuracy: 0.9010
Epoch 20/50
903/909 [============================>.] - ETA: 0s - loss: 0.0636 - accuracy: 0.9841
Epoch 20: val_accuracy did not improve from 0.90105
909/909 [==============================] - 9s 10ms/step - loss: 0.0642 - accuracy: 0.9838 - val_loss: 0.6863 - val_accuracy: 0.9007
Epoch 21/50
906/909 [============================>.] - ETA: 0s - loss: 0.0643 - accuracy: 0.9839
Epoch 21: val_accuracy improved from 0.90105 to 0.90681, saving model to checkpoint/best_model.h5
909/909 [==============================] - 12s 13ms/step - loss: 0.0645 - accuracy: 0.9839 - val_loss: 0.6563 - val_accuracy: 0.9068
Epoch 22/50
907/909 [============================>.] - ETA: 0s - loss: 0.0574 - accuracy: 0.9868
Epoch 22: val_accuracy did not improve from 0.90681
909/909 [==============================] - 10s 11ms/step - loss: 0.0573 - accuracy: 0.9868 - val_loss: 0.6460 - val_accuracy: 0.9033
Epoch 23/50
906/909 [============================>.] - ETA: 0s - loss: 0.0509 - accuracy: 0.9876
Epoch 23: val_accuracy did not improve from 0.90681
Restoring model weights from the end of the best epoch: 13.
909/909 [==============================] - 9s 10ms/step - loss: 0.0510 - accuracy: 0.9875 - val_loss: 0.7883 - val_accuracy: 0.8769
Epoch 23: early stopping
<keras.src.callbacks.History at 0x79b1d0f1d8d0>

It is one of the best results that we got!

So, there's always room for imroving your model, so keep training it until you see good results.


#### Evaluation and Testing: Performance on Unseen Data

**Load the Best Saved Model:**

- Use the model saved during training to ensure evaluation on the best version.

```python
from tensorflow.keras.models import load_model

model = load_model(checkpoint_path)
```

**Evaluation:**

- Evaluate the model's performance on the test set.

```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc}')
```

#### Movie Recommendations: Based on Detected Emotions

**Add an External Movie Dataset:**

- Assume `movie_df` is a DataFrame containing movie titles and genres.

**Implement the Movie Recommendation System:**

1. **Emotion to Genre Mapping:** Create a dictionary mapping detected emotions to movie genres.
2. **Recommendation Function:** Based on the predicted emotion, filter `movie_df` to suggest movies of matching genres.

```python
emotion_to_genre = {
    'anger': 'Action|Thriller',
    'disgust': 'Horror',
    'fear': 'Horror|Thriller',
    'happiness': 'Comedy|Family',
    'sadness': 'Drama',
    'surprise': 'Adventure|Sci-Fi',
    'neutral': 'Documentary'
}

def recommend_movies(emotion):
    genre = emotion_to_genre[emotion]
    recommended = movie_df[movie_df['genres'].str.contains(genre)]
    return recommended.head(5)
```

**Putting It All Together:**

- Predict emotion from an image.
- Use the predicted emotion to recommend movies.

```python
# Assume `predict_emotion(image)` is a function that predicts emotion from an image
predicted_emotion = predict_emotion(image_path)
recommended_movies = recommend_movies(predicted_emotion)
print("Recommended Movies based on your emotion:", recommended_movies)
```

Through these steps, a CNN model is built and trained for emotion recognition, evaluated for its accuracy, and then applied in a practical context to recommend movies based on the detected emotions. This showcases the potential of machine learning models to provide personalized experiences.

**Note:** For the movie recommendation system to work effectively, make sure the `movie.csv` dataset is correctly formatted and placed in the specified directory.

Enjoy exploring emotions and discovering movies that match your mood!

---