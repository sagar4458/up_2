from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def build_model():
    # Load the pre-trained VGG16 model, excluding the top classification layer
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False  # Freeze the layers of VGG16

    # Add custom layers on top of the base model
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=x)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model():
    model = build_model()

    # Data augmentation and normalization for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2  # Split 20% of the data for validation
    )

    # Training data generator
    train_generator = train_datagen.flow_from_directory(
        'preprocessed_data/',  # Directory containing the preprocessed data
        target_size=(224, 224),  # Resize images to 224x224
        batch_size=32,
        class_mode='binary',
        subset='training'  # Use the training subset
    )

    # Validation data generator
    val_generator = train_datagen.flow_from_directory(
        'preprocessed_data/',  # Directory containing the preprocessed data
        target_size=(224, 224),  # Resize images to 224x224
        batch_size=32,
        class_mode='binary',
        subset='validation'  # Use the validation subset
    )

    # Train the model
    model.fit(train_generator, validation_data=val_generator, epochs=10)

    # Save the model in TensorFlow format
    model.save('models/crack_detector.h5')

    # Convert the model to TensorFlow Lite format
    convert_to_tflite(model)


def convert_to_tflite(model):
    # Convert the Keras model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model
    with open('models/crack_detector.tflite', 'wb') as f:
        f.write(tflite_model)


if __name__ == "__main__":
    train_model()
