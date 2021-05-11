from utils.device_util import cur_device
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
import wandb

def eval_accuracy(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(cur_device), y_test.to(cur_device)
            y_val = model(X_test)  # we don't flatten the data this time
            predicted = torch.max(y_val, 1)[1]
            correct += (predicted == y_test).sum()
    # print(f'Test accuracy: {correct.item()}/{len(X_test)} = {correct.item() * 100 / (len(X_test)):7.3f}%')
    return predicted.cpu(), y_test.cpu(), (correct.item()/len(test_loader.dataset) * 100)

def get_confusion_matrix(predicted, y_test):
    # print the matrix
    print("Confusion matrix:")
    np.set_printoptions(formatter=dict(int=lambda x: f'{x:4}'))
    print(np.arange(10).reshape(1, 10))
    print()

    # print the confusion matrix
    con_mat = confusion_matrix(predicted.view(-1), y_test.view(-1))
    print(con_mat)
    print()

    # plot confusion matrix on wandb
    wandb.sklearn.plot_confusion_matrix(y_test, predicted, labels=[x for x in range(10)])

    return con_mat

# Calculate precision
def eval_precision(con_mat):
    tpfp = 0
    tp = 0
    precision = []

    for pred in range(len(con_mat[0])):
        tpfp = 0
        tp = 0
        for true_out in range(len(con_mat)):
            if pred == true_out:
                tp = con_mat[true_out][pred]
            tpfp += con_mat[true_out][pred]
        precision.append(tp / tpfp)
    return precision

# Calculate recall
def eval_recall(con_mat):
    tpfn = 0
    tp = 0
    recall = []

    for true_out in range(len(con_mat)):
        tpfn = 0
        tp = 0
        for pred in range(len(con_mat[0])):
            if true_out == pred:
                tp = con_mat[true_out][pred]
            tpfn += con_mat[true_out][pred]
        recall.append(tp/tpfn)
    return recall

# Calculate F1-score
def eval_f1(precision, recall):
    f1 = []
    for i in range(len(precision)):
        f1.append(2 * ((precision[i]*recall[i]) / (precision[i] + recall[i])))
    return f1

def get_all_results(model, test_load_all):
    # Get final accuracy
    predicted, y_test, acc = eval_accuracy (model, test_load_all)
    print(f'Final Testing Accuracy: {acc:.3f}%.')

    # Get all metrics score
    con_mat = get_confusion_matrix(predicted, y_test)
    precision = eval_precision(con_mat)
    recall = eval_recall(con_mat)
    f1 = eval_f1(precision, recall)

    #Combine into dataframe
    precision = [ '%.3f' % elem for elem in precision]
    recall = [ '%.3f' % elem for elem in recall]
    f1 = [ '%.3f' % elem for elem in f1]
    metrics_table = [precision, recall, f1]

    df = pd.DataFrame (metrics_table,columns=[n for n in range(len(precision))])
    df = df.T
    df.columns = ["Precision", "Recall", "F1_score"]
    print(df)
    return df


