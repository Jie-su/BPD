from model.layers import *


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
    elif type(layer) == nn.Linear:
        layer.weight.data.normal_(0.0, 1e-4)


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(config.input_dim, 64, kernel_size=(5, 1)),
            nn.MaxPool2d((2, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(5, 1)),
            nn.MaxPool2d((2, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 1)),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, config.cls_num),
        )

    def forward(self, x):
        x = x.unsqueeze(dim=3)  # input size: (batch_size, channel, win， 1)
        x = self.cnn(x)
        x = x.reshape(x.size(0), -1)
        feature = x
        x = self.classifier(x)
        return feature, x
        # return x


class ConvLSTMv1(nn.Module):
    def __init__(self, config):
        super(ConvLSTMv1, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(config.input_dim, 64, kernel_size=(5, 1)),
            nn.Conv2d(64, 64, kernel_size=(5, 1)),
            nn.Conv2d(64, 64, kernel_size=(5, 1)),
            nn.Conv2d(64, 64, kernel_size=(5, 1)),
        )

        self.lstm = nn.LSTM(64, 128, 2, batch_first=True)
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(128, config.cls_num)

    def forward(self, x):
        x = x.unsqueeze(dim=3)  # input size: (batch_size, channel, win， 1)
        x = self.features(x)  # [b, 64 , h , w]
        x = x.view(x.shape[0], -1, 64)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = x.view(x.shape[0], 128)
        x = self.classifier(x)

        return x


class ConvLSTMv2(nn.Module):
    def __init__(self, config):
        super(ConvLSTMv2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(config.input_dim, 64, kernel_size=(5, 1)),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 64, kernel_size=(5, 1)),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 32, kernel_size=(3, 1)),
        )

        self.lstm = nn.LSTM(32, 128, 1, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(128, config.cls_num)

    def forward(self, x):
        x = x.unsqueeze(dim=3)  # input size: (batch_size, channel, win， 1)
        x = self.features(x)  # [b, 64 , h , w]
        x = x.view(x.shape[0], -1, 32)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = x.view(x.shape[0], 128)
        x = self.classifier(x)

        return x


class Feature_CNN(nn.Module):
    def __init__(self, config):
        super(Feature_CNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(config.input_dim, 64, kernel_size=(5, 1)),
            nn.MaxPool2d((2, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(5, 1)),
            nn.MaxPool2d((2, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 1)),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.unsqueeze(dim=3)  # input size: (batch_size, channel, win， 1)
        x = self.cnn(x)
        x = x.reshape(x.size(0), -1)
        return x


class Feature_ConvLSTMv2(nn.Module):
    def __init__(self, config):
        super(Feature_ConvLSTMv2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(config.input_dim, 64, kernel_size=(5, 1)),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 64, kernel_size=(5, 1)),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 32, kernel_size=(3, 1)),
        )

        self.lstm = nn.LSTM(32, 128, 1, batch_first=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(dim=3)  # input size: (batch_size, channel, win， 1)
        x = self.features(x)  # [b, 64 , h , w]
        x = x.view(x.shape[0], -1, 32)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = x.view(x.shape[0], 128)

        return x


class Feature_disentangle(nn.Module):
    def __init__(self, config):
        super(Feature_disentangle, self).__init__()
        self.fc1 = nn.Linear(config.output_dim, int(config.output_dim / 4))
        self.bn1_fc = nn.BatchNorm1d(int(config.output_dim / 4))

    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        return x


class Feature_discriminator(nn.Module):
    def __init__(self, config):
        super(Feature_discriminator, self).__init__()
        self.fc1 = nn.Linear(int(config.output_dim / 4), 2)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        return x


class Reconstructor_Net(nn.Module):
    def __init__(self, config):
        super(Reconstructor_Net, self).__init__()
        self.fc = nn.Linear(int(config.output_dim / 2), config.output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x


class Mine_Net(nn.Module):
    def __init__(self, config):
        super(Mine_Net, self).__init__()
        self.fc1_x = nn.Linear(int(config.output_dim / 4), int(config.output_dim / 8))
        self.fc1_y = nn.Linear(int(config.output_dim / 4), int(config.output_dim / 8))
        self.fc2 = nn.Linear(int(config.output_dim / 8), 1)

    def forward(self, x, y):
        h1 = F.leaky_relu(self.fc1_x(x) + self.fc1_y(y))
        h2 = self.fc2(h1)
        return h2


class Predictor_Net(nn.Module):
    def __init__(self, config):
        super(Predictor_Net, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(int(config.output_dim / 4), config.cls_num)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x
