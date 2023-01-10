import numpy as np
import torch
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader 
from src.data.util import MyDataset
from src.models.model import MyAwesomeModel
from sklearn.manifold import TSNE
   
@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('figure_filepath', type=click.Path())

def main(model_filepath, data_filepath, figure_filepath):
    logger = logging.getLogger(__name__)
    logger.info('visualizing')
    model = MyAwesomeModel().float()
    state_dict = torch.load(model_filepath)
    model.load_state_dict(state_dict)
    train_set = torch.load(data_filepath)
    train_dset = MyDataset(train_set['images'], train_set['labels'])
    train_dl = DataLoader(train_dset, batch_size = 64, shuffle = True)

    features = np.zeros((len(train_dset), 8, 20, 20))
    for i, (images, _) in enumerate(train_dl):
        images = images.view(images.shape[0], 1, 28, 28).type(torch.FloatTensor)
        features[64*i:64*(i+1)] = model.backbone(images).detach().numpy()
        
    features = features.reshape(len(features), -1)
    features_reduced = TSNE().fit_transform(features)
    plt.scatter(features_reduced[:, 0], features_reduced[:, 1], s = 1)
    plt.savefig(figure_filepath)
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()