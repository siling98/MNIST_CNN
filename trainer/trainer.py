from utils.device_util import cur_device
import time
import torch
from trainer import evaluate
from utils import file_util
import wandb
wandb.init(project="mnist")

class Trainer:
    def __init__(self, optimizer, run_num, model_path, model_name):
        self.optimizer = optimizer
        self.run_num = run_num
        self.model_path = model_path
        self.model_name = model_name

    def train_one_epoch(self, train_loader, epoch):
        trn_corr = 0

        for b, (X_train, y_train) in enumerate(train_loader):
            b += 1

            X_train, y_train = X_train.to(cur_device), y_train.to(cur_device)
            # Apply the model
            y_pred, loss = self.optimizer.step_optimizer(X_train, y_train)
            # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr

            # Print last batch results after every epoch (training acc and loss)
            if b % 5000 == 0:
                train_acc = trn_corr.item() * 100 / (10 * b)
                print(f'Epoch: {epoch:2}  batch: {b:4} [{10 * b:6}/50000]  loss: {loss.item():10.8f}  \
    accuracy: {trn_corr.item() * 100 / (10 * b):7.3f}%')

                # log to wandb
                wandb.log({"Training Loss": loss, "Epoch": epoch})
                wandb.log({"Training Accuracy": train_acc, "Epoch": epoch})

    def train(self, train_loader, val_loader, epochs):
        start_time = time.time()
        # loop epochs
        for i in range(epochs):
            # Run the training batches / 1 epoch:
            self.train_one_epoch(train_loader, i)

            # Get Test Accuracy per epoch
            predicted, y_test, test_accuracy = evaluate.eval_accuracy(self.optimizer.model, val_loader)
            print(f'Epoch: {i}  Test Accuracy: {test_accuracy:.3f}%')
            wandb.log({"Validation Accuracy": test_accuracy, "Epoch": i})

            # save model for every epoch
            file_util.save_model(self.optimizer.model, self.model_path, self.model_name, i, self.run_num)

        print(f'\nDuration: {time.time() - start_time:.0f} seconds')  # print the time elapsed
