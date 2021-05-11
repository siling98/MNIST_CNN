import os
import torch

def get_run_num(dir):
    path, dirs, files = next(os.walk(dir))
    run_num = len(files) + 1
    return run_num

def save_model(model, dir, model_name, epoch, run_num):
    # Save model under CNN_MNIST/saved_model/run_<run_num>
    dirname = dir + "/" + f'run_{run_num:03}'

    #check if directory exists, if not create new
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    filename = dirname + "/" + model_name + f'_epoch_{epoch:03}.pt'
    torch.save(model.state_dict(), filename)
    print(f'Model is saved under {dirname} as MNISTDatasetModel_epoch_{epoch:03}.pt.')

def save_results(dir, run_num, df_results):
    filename = dir + f'/run_{run_num:03}.csv'
    df_results.to_csv(filename)
    print(f'Saved metrics under results folder as run_{run_num:03}.csv')