import argparse
import sys

import torch
from torch import nn

from data import mnist
from model import MyAwesomeModel

from torch import optim
import matplotlib.pyplot as plt
    
class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=float(args.lr))
        train_dl, _ = mnist()
        train_loss = []
        for e in range(10):
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
        torch.save(model, 'trained_model.pt')
        plt.plot(train_loss)
        plt.xlabel('Training step')
        plt.ylabel('Training loss') 
        plt.savefig('training.png')                          
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = torch.load(args.load_model_from)
        _, test_dl = mnist()
        with torch.no_grad():
            correct = 0
            for images, labels in test_dl:
                log_ps = model(images)
                top_p, top_class = log_ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                correct += equals.sum()
            print(f'Accuracy: {correct/len(test_dl)}%')
                
if __name__ == '__main__':
    TrainOREvaluate()