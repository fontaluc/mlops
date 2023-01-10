import torch
from torch import nn

from model import MyAwesomeModel

from torch import optim
import matplotlib.pyplot as plt

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from torch.utils.data import DataLoader
from src.data.util import MyDataset
   
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path())
@click.argument('figure_filepath', type=click.Path())
@click.option('--lr', type=float)
@click.option('--bs', type=int)
@click.option('--epochs', type=int)
def main(input_filepath, model_filepath, figure_filepath, lr, bs, epochs):
    logger = logging.getLogger(__name__)
    logger.info('training model')
    
    model = MyAwesomeModel().float()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_set = torch.load(input_filepath)
    train_dset = MyDataset(train_set['images'], train_set['labels'])
    train_dl = DataLoader(train_dset, batch_size = bs, shuffle = True)
    train_loss = []
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_dl:
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss = running_loss/len(train_dl)
        train_loss.append(loss)
        print(f'Training loss: {loss}')
    torch.save(model.state_dict(), model_filepath)
    plt.plot(train_loss)
    plt.xlabel('Training step')
    plt.ylabel('Training loss') 
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