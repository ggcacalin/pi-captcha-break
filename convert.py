import tensorflow as tf
import tensorflow.lite as tflite


def main():
    modelname = "baby"
    outputmodel = "baby_lite"

    json_file = open(modelname + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights(modelname + '.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converted_model = converter.convert()

    with open(outputmodel + '.tflite', 'wb') as f:
        f.write(converted_model)

if __name__ == '__main__':
    main()