# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import json

def _concat_content(content):
    images = [file['images'] for file in content]
    images = torch.from_numpy(np.concatenate(images))
    labels = [file['labels'] for file in content]
    labels = torch.from_numpy(np.concatenate(labels))
    return {'images': images, 'labels': labels}

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    train_files = ['train_{}.npz'.format(i) for i in range(5)]
    test_file = ['test.npz']
    
    train_content = [np.load(input_filepath + f) for f in train_files]
    test_content = [np.load(input_filepath + f) for f in test_file]
    
    train_set = _concat_content(train_content)
    test_set =  _concat_content(test_content)
    
    mean = train_set['images'].mean()
    std = train_set['images'].std()
    
    train_set['images'] = (train_set['images'] - mean)/std
    test_set['images'] = (test_set['images'] - mean)/std
    
    example_images = test_set['images'][:10].numpy()
    
    torch.save(train_set, output_filepath + 'train.pt')
    with open(output_filepath + 'example_images.npy', 'wb') as f:
        np.save(f, example_images)
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
