import argparse
import os
import sys
import numpy as np
import h5py
import json
import torch
# from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import imageio
from PIL import Image, ImageFont, ImageDraw


def create_lang_token_input_files(dataset, custom_data, image_folder, es_image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    '''
        Creates input files for training, validation, and test data.
        :param dataset: name of dataset 'coco'
        :param custom_data: path of multilingual captions dataset
        :param image_folder: folder with downloaded images
        :param captions_per_image: number of captions to sample per image
        :param min_word_freq: words with frequency less than this number are removed
        :param output_folder: folder to save files
        :param max_len: maximum length of a caption
        :return: None
    '''

    assert dataset in {'coco'}

    # Read JSON file
    with open(custom_data, 'r') as j:
        data = json.load(j)
    
    # Create a list to store languages
    languages = []

    # Read image paths and captions for each image
    image_data = {
        'train': {
            'paths': [],
            'captions': [],
            'languages': []
        },
        'val': {
            'paths': [],
            'captions': [],
            'languages': []
        },
        'test': {
            'paths': [],
            'captions': [],
            'languages': []
        }
    }

    word_freq = Counter()

    for img in data['images']:
        captions = []
        lang = []
        for c in img['sentences']:
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])
                lang.append(c['language'])
                languages.append(c['language'])

        if len(captions) == 0:
            continue

        # path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
        #     image_folder, img['filename'])
        if 'imgid' in img:
            path = os.path.join(image_folder, img['filepath'], img['filename'])
        else:
            path = os.path.join(es_image_folder, img['filename'])


        split = img['split']
        if split in {'train', 'restval'}:
            split = 'train'

        image_data[split]['paths'].append(path)
        image_data[split]['captions'].append(captions)
        image_data[split]['languages'].append(lang)

    # Print the length of train, val, and test data
    print(len(image_data['train']['paths']))
    print(len(image_data['val']['paths']))
    print(len(image_data['test']['paths']))

    # Sanity check
    assert len(image_data['train']['paths']) == len(image_data['train']['captions']) == len(image_data['train']['languages'])
    assert len(image_data['val']['paths']) == len(image_data['val']['captions']) == len(image_data['val']['languages'])
    assert len(image_data['test']['paths']) == len(image_data['test']['captions']) == len(image_data['test']['languages'])

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<en>'] = len(word_map) + 1 # English token
    word_map['<it>'] = len(word_map) + 1 # Italian token
    word_map['<es>'] = len(word_map) + 1 # Spanish token
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Save word map to a JSON
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)
    print("{} words write into WORDMAP".format(len(word_map)))

    # Create a single HDF5 dataset and JSON files for captions, caption lengths, and languages
    seed(123)
    for split, data in image_data.items():
        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image
            images = h.create_dataset('images', (len(data['paths']), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []
            enc_langs = []

            for i, path in enumerate(tqdm(data['paths'])):
                # Sample captions
                if len(data['captions'][i]) < captions_per_image:
                    captions = data['captions'][i] + [choice(data['captions'][i]) for _ in range(captions_per_image - len(data['captions'][i]))]
                    lang = data['languages'][i] + [choice(data['languages'][i]) for _ in range(captions_per_image - len(data['languages'][i]))]
                else:
                    # captions = sample(data['captions'][i], k=captions_per_image)
                    # lang = sample(data['languages'][i], k=captions_per_image)
                    captions = data['captions'][i][:captions_per_image]
                    lang = data['languages'][i][:captions_per_image]

                # Sanity check
                assert len(captions) == len(lang) == captions_per_image

                # Read images
                img = imageio.imread(data['paths'][i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2) # [256, 256, 1+1+1]
                img = np.array(Image.fromarray(img).resize((256, 256)))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    if lang[j] == 'en':
                        enc_c = [word_map['<en>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
                    elif lang[j] == 'it':
                        enc_c = [word_map['<it>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
                    elif lang[j] == 'es':
                        enc_c = [word_map['<es>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
                    else:
                        print("Language error")
                        sys.exit()
                   
                    enc_captions.append(enc_c)
                    caplens.append(len(c) + 2)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multilingual_Image_Captioning')
    parser.add_argument('--dataset', default="coco", help='Default MSCOCO 14 Dataset.')
    parser.add_argument('--custom_data', default="./multilingual_data.json",
                        help='path of captions dataset.')
    parser.add_argument('--image_folder_es', default="./spanish/MS-COCO-ES/images/",
                         help='path of image dataset.')
    parser.add_argument('--image_folder', default="//coco_2014/", help='path of image dataset.')
    parser.add_argument('--captions_per_image', type=int, default=5, help='How many captions each image has?')
    parser.add_argument('--min_word_freq', type=int, default=3, help='the minimum frequency of words')
    parser.add_argument('--output_folder', default='./data_folder', help='output filepath.')
    parser.add_argument('--max_len', type=int, default=50, help='the maximum length of each caption.')
    args = parser.parse_args()
    
    # Create input files (along with word map)
    '''
        Create input files for training, validation, and test data.
        Save images and captions to hdf5 file, and save captions and caption lengths to json file.
        Create word map as vocabulary.
    '''
    create_lang_token_input_files(dataset=args.dataset,
                                  custom_data=args.custom_data,
                                  image_folder=args.image_folder,
                                  es_image_folder=args.image_folder_es,
                                  captions_per_image=args.captions_per_image,
                                  min_word_freq=args.min_word_freq,
                                  output_folder=args.output_folder,
                                  max_len=args.max_len
                                  )