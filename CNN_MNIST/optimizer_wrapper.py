import torch.optim as optim

class OptimizerWarpper:
    def __init__(self, model, lr):
        self.model = model
        self.opt_obj = optim.Adam(self.model.parameters(), lr)

    def step_optimizer(self, data, label):
        self.opt_obj.zero_grad()
        output = self.model.forward(data)
        loss = self.model.criterion(output, label)
        loss.backward()
        self.opt_obj.step()

        return output, loss