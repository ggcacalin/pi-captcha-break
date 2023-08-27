#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import random2 as random
import tensorflow as tf
import tensorflow.keras as keras

# Build a Keras model given some parameters
def create_model(captcha_length, captcha_num_symbols, input_shape, model_depth=5, module_size=2):
  input_tensor = keras.Input(input_shape)
  x = input_tensor
  for i, module_length in enumerate([module_size] * model_depth):
      for j in range(module_length):
          x = keras.layers.Conv2D(32*2**min(i,3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
          x = keras.layers.BatchNormalization()(x)
          x = keras.layers.Activation('relu')(x)
      x = keras.layers.MaxPooling2D(2)(x)

  x = keras.layers.Flatten()(x)
  x = [keras.layers.Dense(captcha_num_symbols, activation='softmax', name='char_%d'%(i+1))(x) for i in range(captcha_length)]
  model = keras.Model(inputs=input_tensor, outputs=x)

  return model

# A Sequence represents a dataset for training in Keras
# In this case, we have a folder full of images
# Elements of a Sequence are *batches* of images, of some size batch_size
class ImageSequence(keras.utils.Sequence):
    def __init__(self, directory_name, dict_file, batch_size, captcha_length, captcha_symbols, captcha_width, captcha_height):
        self.directory_name = directory_name
        self.batch_size = batch_size
        self.captcha_length = captcha_length
        self.captcha_symbols = captcha_symbols
        self.captcha_width = captcha_width
        self.captcha_height = captcha_height
        self.dict_file = dict_file

        file_list = os.listdir(self.directory_name)
        self.files = dict(zip(map(lambda x: x.split('.')[0], file_list), file_list))
        self.used_files = []
        self.count = len(file_list)

    def __len__(self):
        return int(numpy.floor(self.count / self.batch_size))-1

    def __getitem__(self, idx):
        X = numpy.zeros((self.batch_size, self.captcha_height, self.captcha_width, 3), dtype=numpy.float32)
        y = [numpy.zeros((self.batch_size, len(self.captcha_symbols)), dtype=numpy.uint8) for i in range(self.captcha_length)]

        for i in range(self.batch_size):
            random_image_label = random.choice(list(self.files.keys()))
            random_image_file = self.files[random_image_label]

            # We've used this image now, so we can't repeat it in this iteration
            self.used_files.append(self.files.pop(random_image_label))

            # We have to scale the input pixel values to the range [0, 1] for
            # Keras so we divide by 255 since the image is 8-bit RGB
            raw_data = cv2.imread(os.path.join(self.directory_name, random_image_file))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            processed_data = numpy.array(rgb_data) / 255.0
            X[i] = processed_data

            # Pad with $ to have everything of length 6
            random_image_label = self.dict_file[random_image_label].ljust(self.captcha_length, '$')

            for j, ch in enumerate(random_image_label):
                y[j][i, :] = 0
                y[j][i, self.captcha_symbols.find(ch)] = 1

        return X, y

def main():
    width=128
    height=64
    length=6
    batchsize=128
    training = 'training_set'
    validation = 'validation_set'
    outputmodel = 'baby'
    #change input to load some weights
    inputmodel = None
    epochs=10
    symbols_file=open('symbols_padding.txt', 'r')
    captcha_symbols = symbols_file.readline()

    train_dict = {}
    with open('training_dictionary.txt') as train_dict_file:
        lines = train_dict_file.readlines()
        for line in lines:
            words = line.split(' ')
            train_dict[words[0]] = words[1].replace('\n', '')

    validate_dict = {}
    with open('validation_dictionary.txt') as validate_dict_file:
        lines = validate_dict_file.readlines()
        for line in lines:
            words = line.split(' ')
            validate_dict[words[0]] = words[1].replace('\n', '')

    with tf.device('/device:GPU:1'):
    # with tf.device('/device:CPU:0'):
        model = create_model(length, len(captcha_symbols), (height, width, 3))

        if inputmodel is not None:
            model.load_weights(outputmodel+'_resume.h5')

        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                      metrics=['accuracy'])

        model.summary()

        training_data = ImageSequence(training, train_dict, batchsize, length, captcha_symbols, width, height)
        validation_data = ImageSequence(validation, validate_dict, batchsize, length, captcha_symbols, width, height)

        callbacks = [keras.callbacks.EarlyStopping(patience=3),
                     # keras.callbacks.CSVLogger('log.csv'),
                     keras.callbacks.ModelCheckpoint(outputmodel+'.h5', save_best_only=False)]

        # Save the model architecture to JSON
        with open(outputmodel+".json", "w") as json_file:
            json_file.write(model.to_json())

        try:
            model.fit_generator(generator=training_data,
                                validation_data=validation_data,
                                epochs=epochs,
                                callbacks=callbacks,
                                use_multiprocessing=True)
            
        except KeyboardInterrupt:
            print('KeyboardInterrupt caught, saving current weights as ' + outputmodel+'_resume.h5')
            model.save_weights(outputmodel+'_resume.h5')

if __name__ == '__main__':
    main()
