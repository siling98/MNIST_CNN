from utils.device_util import cur_device
from CNN_MNIST.cnn_mnist import ConvolutionalNetwork
from trainer.trainer import Trainer
from trainer import evaluate
from data_handler.data_loader import MnistDataLoader
from config.parameters import ModelConfig
from CNN_MNIST.optimizer_wrapper import OptimizerWarpper
from utils import file_util

if __name__ == '__main__':
    data = MnistDataLoader(ModelConfig.train_batch_size, ModelConfig.test_batch_size, ModelConfig.data_dir)
    model = ConvolutionalNetwork()
    model.to(cur_device)
    optimizer = OptimizerWarpper(model, ModelConfig.lr)

    # Get the run number:
    run_num = file_util.get_run_num(ModelConfig.save_results_dir)

    # Define trainer object class and Train the model
    train_model = Trainer(optimizer, run_num, ModelConfig.save_model_dir, ModelConfig.save_model_name)
    train_model.train(data.train_loader, data.val_loader, ModelConfig.epochs)

    # Get all results and save:
    df_results = evaluate.get_all_results(optimizer.model, data.test_loader)
    file_util.save_results(ModelConfig.save_results_dir, run_num, df_results)

