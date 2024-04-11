import torch
import numpy as np
from matplotlib import pyplot as plt

"""
Implements early stopping to halt the training process when validation loss ceases to decrease, preventing overfitting. 
It monitors validation loss and stops training if no improvement is observed after a specified number of epochs, 
defined by 'patience'. Additionally, it saves the model at its best performance state.
"""


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


"""
calculate the accuracy
"""


def calculate_accuracy(preds, labels):
    _, preds_max_indices = torch.max(preds, dim=1)
    correct_preds = (preds_max_indices == labels).float()
    accuracy = correct_preds.mean().item()
    return accuracy


"""
Basic training loop for pytorch
"""


def train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs=25, patience=7):
    early_stopping = EarlyStopping(patience=patience, verbose=True, path='model_Kuzushiji.pt')

    train_losses = []
    val_losses = []
    train_accuracy = []
    val_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_train_accuracy = []  # To store accuracy for each batch

        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs)
            acc = calculate_accuracy(preds, labels)
            batch_train_accuracy.append(acc)  # Append accuracy for each batch

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = sum(batch_train_accuracy) / len(batch_train_accuracy)  # Calculate average accuracy for the epoch

        model.eval()
        val_loss = 0.0
        batch_val_accuracy = []  # To store accuracy for each batch in validation

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.cuda(), labels.cuda()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs)
                acc = calculate_accuracy(preds, labels)
                batch_val_accuracy.append(acc)  # Append accuracy for each validation batch

        val_loss = val_loss / len(validation_loader.dataset)
        val_acc = sum(batch_val_accuracy) / len(batch_val_accuracy)  # Calculate average accuracy for validation

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        train_accuracy.append(epoch_acc)  # Append average accuracy of the epoch
        val_accuracy.append(val_acc)  # Append average accuracy of validation

        print(
            f'Epoch {epoch + 1}, Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    plot_training_history(train_losses, val_losses, train_accuracy, val_accuracy)
    model.load_state_dict(torch.load('model_Kuzushiji.pt'))
    return model, train_losses, val_losses, train_accuracy, val_accuracy



def plot_training_history(train_losses, val_losses, train_accuracy, val_accuracy):
    # Plotting training and validation losses
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    # Save the figure
    plt.savefig('training_history.png', dpi=300)
    #plt.show()
