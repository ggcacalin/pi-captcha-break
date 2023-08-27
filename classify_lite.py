import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
import argparse
import os
import time


def decode(characters, y):
    y = np.argmax(np.array(y), axis=2)[:, 0]
    result = ''.join([characters[x] for x in y])
    result = result.replace('$','')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='TFLite model name, without extension', type=str)
    parser.add_argument('--captcha-dir', help='Test set directory', type=str)
    parser.add_argument('--output', help='Prediction storage file', type=str)
    parser.add_argument('--symbols', help='Available characters for classification', type=str)
    parser.add_argument('--shortname', help='Name on first row', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Give model without extension")
        exit(1)

    if args.captcha_dir is None:
        print("Give test set directory")
        exit(1)

    if args.output is None:
        print("Give name of .txt where predictions are stored")
        exit(1)

    if args.symbols is None:
        print("Give character set")
        exit(1)

    if args.shortname is None:
        print('Give shortname for submission')
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")
    # create tflite interpreter
    interpreter = tflite.Interpreter(model_path=args.model_name + '.tflite')
    interpreter.allocate_tensors()
    with open(args.output, 'w') as output_file:
        output_file.write(args.shortname + '\n')
        # shortcuts to interpreter input/output
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        count = 0
        starting_time = time.time()
        for x in sorted(os.listdir(args.captcha_dir)):
            # preprocessing
            raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            image = np.array(rgb_data, dtype=np.float32) / 255.0
            (c, h, w) = image.shape
            image = image.reshape([-1, c, h, w])
            # set image as input tensor and engage interpreter
            interpreter.set_tensor(input_details[0]['index'], image)
            interpreter.invoke()
            # after getting the output characters in the right order through index of details array,
            # assemble copies of tensors for full label prediction
            predictions = [interpreter.get_tensor(dict['index']) for dict in
                           sorted(output_details, key=lambda i: i['index'])]
            output_file.write(x + "," + decode(captcha_symbols, predictions) + "\n")

            print('Classified ' + x)
            count += 1
        print('Classifying ' + str(count) + ' images took ' + str(time.time() - starting_time) + ' seconds')

if __name__ == '__main__':
    main()
