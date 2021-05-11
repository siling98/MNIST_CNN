class BaseConfig:
    image_size = 28
    data_dir = 'Data'
    save_model_dir = 'CNN_MNIST/saved_model'
    save_model_name = 'MNISTDatasetModel'
    save_results_dir = 'results'

class ModelConfig(BaseConfig):
    train_batch_size = 10
    test_batch_size = 10
    lr = 0.001
    epochs = 10

