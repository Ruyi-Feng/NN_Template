from model.model import Net
from torch.utils.data import DataLoader
from torch import optim


class Exp_Main:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 这只是一种device指定的方式，多gpu还可以指定device编号，分布式计算可以采用local_rank的形式指定
        self.loss_func = nn.BCELoss()
        self._build_model()

    def _build_model(self):
        self.model = Net().float().to(self.device)

    def _get_data(self):
        dataset = My_Dataset()
        dataloader = DataLoader(dataset, batch_size=1)
        return dataset, dataloader

    def _select_optimizer(self):
        # 设置optm 必选
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        # 设置scheduler 可选
        lamda1 = lambda step: 1 / np.sqrt(max(step , self.WARMUP))
        scheduler = optim.lr_scheduler.LambdaLR(model_optim, lr_lambda=lamda1, last_epoch=-1)
        return model_optim, scheduler

    def _criterion(self, output, data):
        return self.loss_func(output, data)

    def _save_model(self, epoch):
        torch.save(self.model.state_dict(), 'model_{}.pth'.format(epoch))

    def train(self):
        _, train_loader = self._get_data()
        model_optim, scheduler = self._select_optimizer()
        for epoch in range(total_epoch):
            self.model.train()
            for i, data in enumerate(train_loader):
                model_optim.zero_grad()
                output = self.model(data)
                loss = self._criterion(output, data)
                loss.backward()
                model_optim.step()
                scheduler.step()
            self._save_model()
