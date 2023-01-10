# TUWIEN - WS2022 CV: Task4 - Mask Classification using CNN
# *********+++++++++*******++++INSERT GROUP NO. HERE
import os
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from my_model import MaskClassifier
from my_datamodule import DataModule


class Trainer:

    def __init__(self, model: MaskClassifier, datamodule: DataModule, gpu=False):
        """
        Creates the Trainer
        model: pytorch model
        datamodule: training, validation, test data
        gpu: if the model should be trained on gpu or cpu
        """

        self.model = model
        self.datamodule = datamodule
        self.gpu = gpu
        self.criterion = torch.nn.BCELoss()
        self.best_acc = 0
        self.history = {}

    def fit(self, epochs: int=10, lr: float=1e-4):
        """
        trains the model on the training dataset
        epochs: the number of passes of the eintire training dataset
        lr: learning rate
        """

        self.clear_logging()  # clean history if fit was already called

        if self.gpu:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()  # set model in train state (e.g. dropouts active)

            pbar = tqdm(self.datamodule.train_dataloader())
            pbar.set_description('Epoch {}'.format(epoch))
            losses = []
            accs = []
            for imgs, labels in pbar:
                if self.gpu:
                    imgs = imgs.cuda()
                    labels = labels.cuda()

                imgs.requires_grad = True
                preds = self.model(imgs)

                loss = self.criterion(preds, labels)
                batch_accuracy = (torch.round(preds) ==
                                  labels).sum() / preds.size(0)

                loss = loss.cpu()
                batch_accuracy = batch_accuracy.cpu()

                losses.append(loss.detach().numpy())

                # accuracy
                accs.append(batch_accuracy.detach().numpy())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss, epoch_acc = np.mean(losses), np.mean(accs)
            self.log("train_loss", epoch_loss)
            self.log("train_accuracy", epoch_acc)
            loss, acc = self.validate()
            print(f'Epoch {epoch} Training: Loss: {epoch_loss} Accuracy: {epoch_acc}\n' +
                  f'Epoch {epoch} Validation: Loss: {loss} Accuracy: {acc}')
            if self.best_acc < acc:
                self.save_model()

    def _eval(self, loader, name=str()):
        """
        Evaluates the model using the loader
        loader: data for evaluation
        name: name under which the results are stored
        """

        self.model.eval()
        accs, losses = [], []
        for imgs, labels in loader:
            if self.gpu:
                imgs = imgs.cuda()
                labels = labels.cuda()

            preds = self.model(imgs)
            loss = self.criterion(preds, labels)
            acc = (torch.round(preds) == labels).sum() / preds.size(0)

            loss = loss.cpu()
            acc = acc.cpu()

            losses.append(loss.detach().numpy())
            accs.append(acc.detach().numpy())

        loss, acc = np.mean(losses), np.mean(accs)
        self.log(f'{name}_accuracy', np.mean(accs))
        self.log(f'{name}_loss', np.mean(losses))
        return loss, acc

    def validate(self):
        valloader = self.datamodule.val_dataloader()
        return self._eval(valloader, 'validation')

    def test(self, best_model=True):
        testloader = self.datamodule.test_dataloader()
        if best_model:
            self.load_model()
        return self._eval(testloader, 'test')

    def log(self, key, val):
        # stores the results of eval
        if self.history.get(key, None) is None:
            self.history[key] = []

        self.history[key].append(val)

    def clear_logging(self):
        # clears the result
        self.history = {}

    def plot_performance(self, name: str='', group_no=0):
        """
        Visualizes the performance of training
        name: the name of the visualization
        group_no: your group number
        """

        self.test()
        test_loss = self.history['test_loss']
        test_acc = self.history['test_accuracy']

        train_acc = self.history['train_accuracy']
        val_acc = self.history['validation_accuracy']

        train_loss = self.history['train_loss']
        val_loss = self.history['validation_loss']

        max_acc = np.argmax(val_acc)
        epochs = range(1, len(val_acc)+1)

        fig = plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_acc, label='Training Accuracy')
        plt.plot(epochs, val_acc, label='Validation Accuracy')
        plt.vlines(max_acc, 0, val_acc[max_acc],
                   linestyles="dotted", colors="r")
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.vlines(max_acc, 0, val_loss[max_acc],
                   linestyles="dotted", colors="r")
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        fig.suptitle(str(group_no)+": test accuracy="+str(test_acc) +
                     ", loss: "+str(test_loss), fontsize=14, y=1)
        if name is not None and group_no is not None:
            plt.savefig(os.path.join(os.getcwd(), 'results', name))

        plt.plot()

    def _get_path(self):
        path = f'results/best/{self.model.name}.pth'
        return path

    def save_model(self):
        path = self._get_path()
        torch.save(self.model.state_dict(), path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self._get_path()))

    def predict(self, x, best_model=True):
        """
        preicts a set of samples
        x: tensor of data samples
        best_model: if true the best model is loaded
        """

        if best_model:
            self.load_model()
        self.model.eval()
        return self.model(x).round().detach().numpy()
