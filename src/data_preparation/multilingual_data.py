import json
import random

def combine_dataset(spanish_data, en_it_data, seed):
    print("Shuffel Spanish and Italian dataset to perform Multi-lingual Image Captioning")

    root = {}

    # mix the dataset
    data = spanish_data['images'] + en_it_data['images']
    random.seed(seed)
    random.shuffle(data)
    root['images'] = data

    # save in json file
    json.dump(root, open('multilingual_data_En+It+Es.json', 'w'))
    


if __name__=="__main__":
    spanish_data = json.load(open("./coco_dataset_es_final.json", 'r'))
    en_it_data = json.load(open("./multilingual_data_En+It.json", 'r'))

    combine_dataset(spanish_data, en_it_data, seed=123)