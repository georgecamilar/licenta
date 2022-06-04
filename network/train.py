import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


def create_new_model(mobilenet):
    # prepare for transfer learning
    # freeze weights for model
    mobilenet.trainable = False
    return tf.keras.Sequential([
        mobilenet,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dense(51, activation='softmax')
    ])


def train_model():
    mobilenet_v2 = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False)
    model = create_new_model(mobilenet_v2)
    # Created model
    model.summary()
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest',
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
        validation_split=0.2
    )

    train = train_datagen.flow_from_directory(
        "/Users/georgecamilar/Documents/transferLearning/gt_db",
        target_size=(224, 224),
        batch_size=32,
        shuffle=True,
        class_mode='categorical',
        subset='training')

    validation = train_datagen.flow_from_directory(
        "/Users/georgecamilar/Documents/transferLearning/gt_db",
        target_size=(224, 224),
        batch_size=32,
        shuffle=True,
        class_mode='categorical',
        subset='validation')

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Train model
    model.fit_generator(train,
                        steps_per_epoch=train.samples // 32,
                        validation_data=validation,
                        validation_steps=validation.samples // 32,
                        epochs=25)


def save(model):
    model.save("./weights")
