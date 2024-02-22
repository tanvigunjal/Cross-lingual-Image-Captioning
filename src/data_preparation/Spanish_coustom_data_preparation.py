import os
import sys
import pandas as pd
import nltk
import json
nltk.download('punkt')

def group_captions_by_image_id(dataframe):
    # Initialize an empty dictionary to store grouped captions
    grouped_captions = {}

    for index, row in dataframe.iterrows():
        image_id = row['image_id']
        caption = row['caption']
        split = row['split']
      
        # Check if the image ID is already in the dictionary
        if image_id in grouped_captions:
            grouped_captions[image_id].append({'tokens': tokenization(caption), 
                                               'raw': f'string"{caption}"',
                                                'imgid': image_id,
                                                'language': 'es',
                                                'split': split
                                               })
        else:
            grouped_captions[image_id] = [{'tokens': tokenization(caption), 
                                           'raw': f'string"{caption}"', 
                                           'imgid': image_id,
                                           'language': 'es',
                                           'split': split
                                           }]
            
    return grouped_captions

def convert_to_desired_format(grouped_captions):
    data_list = []
    root = {}

    for image_id, captions in grouped_captions.items():
        file_name = f'{image_id:012d}.jpg'

        data_entry = {
            'filepath': 'train2014',
            'filename': file_name,
            'cocoid': image_id,
            'sentences': captions,
            'lang': 'es'
        }

        data_list.append(data_entry)

        # add split in data_list: train, val, test from captions 
        # add split: train, val, test from captions
        for caption in captions:
            if caption['split'] == 'train':
                data_entry['split'] = caption['split']
            elif caption['split'] == 'val':
                data_entry['split'] = caption['split']
            elif caption['split'] == 'test':
                data_entry['split'] = caption['split']
            else:
               data_entry['split'] = 'train'

    root["images"] = data_list
    
    # save in json format
    with open('coco_dataset_es_final.json', 'w') as fp:
        json.dump(root, fp)

    return data_list


def tokenization(captions):
    tokens = nltk.tokenize.word_tokenize(captions.lower())
    # Remove . , ; "" '' ? !
    tokens = [w for w in tokens if not (w in ['.', ',', ';', '\'', '\"', '?', '!', '``', '\'\''])]
    return tokens

if __name__ == "__main__":
    # Read the dataset
    dataset1 = pd.read_excel("./spanish/MS-COCO-ES/data/train_human_spanish.xlsx")
    dataset2 = pd.read_excel("./spanish/MS-COCO-ES/data/train_machine_spanish.xlsx")

    # add a new column to each dataframe
    dataset1['split'] = 'train'

    # add the two dataframes
    dataset = dataset1.append(dataset2, ignore_index=True)

    # validation dataset
    val_dataset = pd.read_excel("./spanish/MS-COCO-ES/data/validation.xlsx")
    val_dataset['split'] = 'val'

    dataset = dataset.append(val_dataset, ignore_index=True)

    # test dataset
    test_dataset = pd.read_excel("./spanish/MS-COCO-ES/data/test.xlsx")
    test_dataset['split'] = 'test'

    dataset = dataset.append(test_dataset, ignore_index=True)
    result = group_captions_by_image_id(dataset)
    convert_to_desired_format(result)