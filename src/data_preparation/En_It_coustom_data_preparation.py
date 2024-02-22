import os
import sys
import json
import random
import nltk
nltk.download('punkt')

def create_dataset(train_annotations, val_annotations, ):
    print("Customize COCO-2014 english dataset")

    # Combine train and val annotations
    annots = val_annotations['annotations'] + train_annotations['annotations']

    # Combine train and val images
    imgs = val_annotations['images'] + train_annotations['images']

    root = {}

    annotation_id = {}
    for a in annots:
        img_id = a['image_id']
        if img_id not in annotation_id:
            annotation_id[img_id] = []
        annotation_id[img_id].append(a)

    out = []
    for i, img in enumerate(imgs):
        imgid = img['id']
        img_location = 'train2014' if 'train' in img['file_name'] else 'val2014'
        
        img_dict = {}
        img_dict['filepath'] = img_location
        img_dict['filename'] = os.path.join(img['file_name'])
        img_dict['split'] = 'train' if 'train' in img['file_name'] else 'val'
        img_dict['cocoid'] = imgid
        img_dict['imgid'] = int(i)
        img_dict['sentences'] = []  # Initialize sentences list
        img_dict['sentids'] = []   # Initialize sentids list

        sentences = []
        sentids = []
        if imgid in annotation_id:
            for a in annotation_id[imgid]:
                sentence_dict = {
                    "tokens": tokenization(a['caption']),  # Tokenize the caption using NLTK
                    "raw": f'{a["caption"]}', # add raw caption
                    "imgid": img_dict['imgid'], 
                    "sentid": int(a['id']),
                    "language": "en" # add language token
                }
                sentences.append(sentence_dict)
                sentids.append(int(a['id']))
            
            img_dict['sentences'] = sentences
            img_dict['sentids'] = sentids
        else:
            print(f'Image id not found in annotations: {imgid}')
        out.append(img_dict)

    val_count = 0
    test_count = 0
    for item in out:
        if item['filepath'] == 'val2014':
            if val_count < 5000:
                item['split'] = 'val'
                val_count += 1
            elif test_count < 5000:
                item['split'] = 'test'
                test_count += 1
            else:
                item['split'] = 'restval'

    root['images'] = out

    # save out in a json file
    json.dump(root, open('./en_dataset_coco.json', 'w'))

    
def tokenization(captions):
    tokens = nltk.tokenize.word_tokenize(captions.lower())
    # remove . , ; "" '' ? !
    tokens = [w for w in tokens if not (w in ['.', ',', ';', '\'', '\"', '?', '!', '``', '\'\''])]
    return tokens

def shuffel_dataset(en_coco, it_coco, seed):
    print("Shuffel English and Italian dataset to perform Multi-lingual Image Captioning")

    # load the dataset
    en_data = json.load(open(en_coco, 'r'))
    it_data = json.load(open(it_coco, 'r'))

    root = {}

    # mix the dataset
    data = en_data['images'] + it_data['images']
    random.seed(seed)
    random.shuffle(data)
    root['images'] = data

    # save in json file
    json.dump(root, open('multilingual_data_En+It.json', 'w'))

if __name__ == '__main__':
    # load annotations
    en_train_annotations = json.load(open('./coco_2014/annotations/captions_train2014.json', 'r'))
    en_val_annotations = json.load(open('./coco_2014/annotations/captions_val2014.json', 'r'))
    it_train_annotations = json.load(open('./italian/captions_ita_trainingset_train2014.json', 'r'))
    it_val_annotations = json.load(open('./italian/captions_ita_trainingset_val2014.json', 'r'))
    
    seed = 42
    
    '''
       Customize the data to Karpathy splits for Image Captioning
       for both English and Italian datasets
       and save "en_dataset_coco.json" and "it_dataset_coco.json"
    '''
    create_dataset(en_train_annotations, en_val_annotations)


    '''
         Shuffel English and Italian dataset to perform Multi-lingual Image Captioning
         and save "multilingual_data.json"
    '''
    en_coco = "./en_dataset_coco.json"
    it_coco = "./it_dataset_coco.json"    
    shuffel_dataset(en_coco, it_coco, seed)