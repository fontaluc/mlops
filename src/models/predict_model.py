import numpy as np
import torch
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.models.model import MyAwesomeModel
   
@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('data_filepath', type=click.Path(exists=True))

def main(model_filepath, data_filepath):
    logger = logging.getLogger(__name__)
    logger.info('predicting')
    
    model = MyAwesomeModel().float()
    state_dict = torch.load(model_filepath)
    model.load_state_dict(state_dict)
    with open(data_filepath, 'rb') as f:
        images = torch.from_numpy(np.load(f))
    with torch.no_grad():
        log_ps = model(images)
        top_p, top_class = log_ps.topk(1, dim=1)
        print(f'Predictions: {top_class}')
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()