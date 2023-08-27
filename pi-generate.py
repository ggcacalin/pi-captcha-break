import os
import numpy
import random2 as random
import cv2
import captcha.image
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Image width', type=int)
    parser.add_argument('--height', help='Image height', type=int)
    parser.add_argument('--length', help='Largest possible number of characters', type=int)
    parser.add_argument('--count', help='No. captchas of one length value', type=int)
    parser.add_argument('--outputdir', help='Captcha storage directory', type=str)
    parser.add_argument('--symbols', help='Character set', type=str)
    parser.add_argument('--dictionary', help = 'Dictionary linking file names to image labels', type=str)
    args = parser.parse_args()

    if args.width is None:
        print("Give width")
        exit(1)

    if args.height is None:
        print("Give height")
        exit(1)

    if args.length is None:
        print("Give length")
        exit(1)

    if args.count is None:
        print("Give captcha count")
        exit(1)

    if args.outputdir is None:
        print("Give storage directory")
        exit(1)

    if args.symbols is None:
        print("Give character set")
        exit(1)

    captcha_generator = captcha.image.ImageCaptcha(width=args.width, height=args.height)
    character_set = open(args.symbols, 'r')

    captcha_symbols = character_set.readline().strip()
    character_set.close()

    print("Generating captchas with symbol set {" + captcha_symbols + "}")

    if not os.path.exists(args.outputdir):
        print("Creating output directory " + args.outputdir)
        os.makedirs(args.outputdir)
    dict = {}
    for l in range(1,args.length+1):
        print(l)
        for i in range(args.count):
            random_str = ''.join([random.choice(captcha_symbols) for j in range(l)])

            image_path = os.path.join(args.outputdir, str((l-1)*args.count+i)+'.png')
            dict[(l-1)*args.count+i] = random_str

            image = numpy.array(captcha_generator.generate_image(random_str))
            cv2.imwrite(image_path, image)
    with open(args.dictionary, 'w') as dict_file:
        for index in dict:
            dict_file.write(str(index) + " " + dict[index] + '\n')

    if __name__ == '__main__':
        main()