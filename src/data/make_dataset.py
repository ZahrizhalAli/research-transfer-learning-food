import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_val_generators(DATA_DIR, SPLIT_SIZE, NUM_BATCHES):
    """

    :param DATA_DIR: Directory of the datasets was taken from (string)
    :param SPLIT_SIZE: Split size of test set. (float)
    :param NUM_BATCHES: Amount of batch size
    :return:
    """
    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=SPLIT_SIZE)
    train_generator = datagen.flow_from_directory(directory=DATA_DIR,
                                                  batch_size=NUM_BATCHES,
                                                  class_mode='categorical',
                                                  target_size=(300, 300),
                                                  subset='training',
                                                    shuffle = True,

    )

    validation_generator = datagen.flow_from_directory(directory=DATA_DIR,
                                                       subset='validation',
                                                       batch_size=NUM_BATCHES,
                                                       class_mode='categorical',
                                                       shuffle=True,
                                                       target_size=(300, 300))

    return train_generator, validation_generator

