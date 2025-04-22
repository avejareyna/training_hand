from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from pathlib import Path

# Rutas
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = BASE_DIR / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Configuración
n_clases = 2
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 25
INPUT_SHAPE = (128, 128, 3)  # Imágenes en color


def training():
    # Preprocesamiento y generación
    datagen_train = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    datagen_val_test = ImageDataGenerator(rescale=1./255)

    train_generator = datagen_train.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = datagen_val_test.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = datagen_val_test.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    filtros = 16
    regularizers_w = 1e-4

    # Modelo
    model = Sequential()
    model.add(Conv2D(filtros, (3, 3), padding='valid', kernel_regularizer=regularizers.l2(regularizers_w), input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))

    model.add(Conv2D(filtros*2, (3, 3), padding='valid', kernel_regularizer=regularizers.l2(regularizers_w)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPool2D(2,2))

    #model.add(Conv2D(filtros*2, (3, 3), padding='valid', kernel_regularizer=regularizers.l2(regularizers_w)))
    #model.add(Dropout(0.5))
    #model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(n_clases, activation='softmax'))

    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint_path = MODELS_DIR / "best_model.keras"

    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[ early_stop,checkpoint]
    )

    # Plots
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("[✓] Entrenamiento completado.")
    print(f"[✓] Mejor modelo guardado en: {checkpoint_path}")