"""
Examples of network architecture.
"""

import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_, uniform_
import torch.nn.functional as F
import gym


def mini_weight_init(m):
    if m.__class__.__name__ == 'Linear':
        m.weight.data.copy_(uniform_(m.weight.data, -3e-3, 3e-3))
        m.bias.data.fill_(0)


def weight_init(m):
    if m.__class__.__name__ == 'Linear':
        m.weight.data.copy_(kaiming_uniform_(m.weight.data))
        m.bias.data.fill_(0)


class PolNet(nn.Module):
    def __init__(self, observation_space, action_space, h1=200, h2=100, deterministic=False):
        super(PolNet, self).__init__()

        self.deterministic = deterministic

        if isinstance(action_space, gym.spaces.Box):
            self.discrete = False
        else:
            self.discrete = True
            if isinstance(action_space, gym.spaces.MultiDiscrete):
                self.multi = True
            else:
                self.multi = False

        self.fc1 = nn.Linear(observation_space.shape[0], h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc1.apply(weight_init)
        self.fc2.apply(weight_init)

        if not self.discrete:
            self.mean_layer = nn.Linear(h2, action_space.shape[0])
            if not self.deterministic:
                self.log_std_param = nn.Parameter(
                    torch.randn(action_space.shape[0])*1e-10 - 1)
            self.mean_layer.apply(mini_weight_init)
        else:
            if self.multi:
                self.output_layers = nn.ModuleList(
                    [nn.Linear(h2, vec) for vec in action_space.nvec])
                list(map(lambda x: x.apply(mini_weight_init), self.output_layers))
            else:
                self.output_layer = nn.Linear(h2, action_space.n)
                self.output_layer.apply(mini_weight_init)

    def forward(self, ob):
        h = F.relu(self.fc1(ob))
        h = F.relu(self.fc2(h))
        if not self.discrete:
            mean = torch.tanh(self.mean_layer(h))
            if not self.deterministic:
                log_std = self.log_std_param.expand_as(mean)
                return mean, log_std
            else:
                return mean
        else:
            if self.multi:
                return torch.cat([torch.softmax(ol(h), dim=-1).unsqueeze(-2) for ol in self.output_layers], dim=-2)
            else:
                return torch.softmax(self.output_layer(h), dim=-1)


class VNet(nn.Module):
    def __init__(self, observation_space, h1=200, h2=100):
        super(VNet, self).__init__()
        self.fc1 = nn.Linear(observation_space.shape[0], h1)
        self.fc2 = nn.Linear(h1, h2)
        self.output_layer = nn.Linear(h2, 1)
        self.apply(weight_init)

    def forward(self, ob):
        h = F.relu(self.fc1(ob))
        h = F.relu(self.fc2(h))
        return self.output_layer(h)


class QNet(nn.Module):
    def __init__(self, observation_space, action_space, h1=300, h2=400):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(observation_space.shape[0], h1)
        if isinstance(action_space, gym.spaces.Box):
            self.fc2 = nn.Linear(action_space.shape[0] + h1, h2)
        else:
            self.fc2 = nn.Linear(action_space.n + h1, h2)
        self.output_layer = nn.Linear(h2, 1)
        self.fc1.apply(weight_init)
        self.fc2.apply(weight_init)
        self.output_layer.apply(mini_weight_init)

    def forward(self, ob, ac):
        h = F.relu(self.fc1(ob))
        h = torch.cat([h, ac], dim=-1)
        h = F.relu(self.fc2(h))
        return self.output_layer(h)


class QNetCNN(nn.Module):
    def __init__(self, observation_space, action_space):
        super(QNetCNN, self).__init__()
        # conv
        self.conv1 = nn.Conv2d(3, 64, kernel_size=6, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # pool
        self.pool1 = nn.MaxPool2d(3, stride=2)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        # fc
        self.fc1 = nn.Linear(action_space.shape[0], 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, ob, ac):
        # observation net
        ob = F.relu(self.conv1(ob))
        ob = self.pool1(ob)
        for i in range(6):
            ob = F.relu(self.conv2(ob))
        ob = self.pool1(ob)
        # action net
        ac = F.relu(self.fc1(ac))
        ac = F.relu(self.fc2(ac))
        ac = ac.view(-1, 1, 1, 64)
        # concat
        h = torch.cat([ob, ac], dim=-1)
        for i in range(6):
            h = F.relu(self.conv3(h))
        h = self.pool2(h)
        for i in range(3):
            h = F.relu(self.conv3(h))
        h = h.view(-1, 64)
        h = F.relu(self.fc3(h))
        out = F.sigmoid(self.output_layer(h))

        return out


class ModelNet(nn.Module):
    def __init__(self, observation_space, action_space, h1=200, h2=200):
        super(ModelNet, self).__init__()
        self.fc1 = nn.Linear(
            observation_space.shape[0] + action_space.shape[0], h1)
        self.fc2 = nn.Linear(h1, h2)
        self.output_layer = nn.Linear(h2, observation_space.shape[0])
        self.fc1.apply(weight_init)
        self.fc2.apply(weight_init)
        self.output_layer.apply(weight_init)

    def forward(self, ob, ac):
        h = torch.cat([ob, ac], dim=-1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.output_layer(h)


class PolNetLSTM(nn.Module):
    def __init__(self, observation_space, action_space, h_size=1024, cell_size=512):
        super(PolNetLSTM, self).__init__()
        self.h_size = h_size
        self.cell_size = cell_size
        self.rnn = True

        if isinstance(action_space, gym.spaces.Box):
            self.discrete = False
        else:
            self.discrete = True
            if isinstance(action_space, gym.spaces.MultiDiscrete):
                self.multi = True
            else:
                self.multi = False

        self.input_layer = nn.Linear(observation_space.shape[0], self.h_size)
        self.cell = nn.LSTMCell(self.h_size, hidden_size=self.cell_size)
        if not self.discrete:
            self.mean_layer = nn.Linear(self.cell_size, action_space.shape[0])
            self.log_std_param = nn.Parameter(
                torch.randn(action_space.shape[0])*1e-10 - 1)

            self.mean_layer.apply(mini_weight_init)
        else:
            if self.multi:
                self.output_layers = nn.ModuleList(
                    [nn.Linear(self.cell_size, vec) for vec in action_space.nvec])
                list(map(lambda x: x.apply(mini_weight_init), self.output_layers))
            else:
                self.output_layer = nn.Linear(self.cell_size, action_space.n)
                self.output_layer.apply(mini_weight_init)

    def init_hs(self, batch_size=1):
        new_hs = (next(self.parameters()).new(batch_size, self.cell_size).zero_(), next(
            self.parameters()).new(batch_size, self.cell_size).zero_())
        return new_hs

    def forward(self, xs, hs, h_masks):
        time_seq, batch_size, *_ = xs.shape

        hs = (hs[0].reshape(batch_size, self.cell_size),
              hs[1].reshape(batch_size, self.cell_size))

        xs = torch.relu(self.input_layer(xs))

        hiddens = []
        for x, mask in zip(xs, h_masks):
            hs = (hs[0] * (1 - mask), hs[1] * (1 - mask))
            hs = self.cell(x, hs)
            hiddens.append(hs[0])
        hiddens = torch.cat([h.unsqueeze(0) for h in hiddens], dim=0)

        if not self.discrete:
            means = torch.tanh(self.mean_layer(hiddens))
            log_std = self.log_std_param.expand_as(means)
            return means, log_std, hs
        else:
            if self.multi:
                return torch.cat([torch.softmax(ol(hiddens), dim=-1).unsqueeze(-2) for ol in self.output_layers], dim=-2), hs
            else:
                return torch.softmax(self.output_layer(hiddens), dim=-1), hs


class VNetLSTM(nn.Module):
    def __init__(self, observation_space, h_size=1024, cell_size=512):
        super(VNetLSTM, self).__init__()
        self.h_size = h_size
        self.cell_size = cell_size
        self.rnn = True

        self.input_layer = nn.Linear(observation_space.shape[0], self.h_size)
        self.cell = nn.LSTMCell(self.h_size, hidden_size=self.cell_size)
        self.output_layer = nn.Linear(self.cell_size, 1)

        self.output_layer.apply(mini_weight_init)

    def init_hs(self, batch_size=1):
        new_hs = (next(self.parameters()).new(batch_size, self.cell_size).zero_(), next(
            self.parameters()).new(batch_size, self.cell_size).zero_())
        return new_hs

    def forward(self, xs, hs, h_masks):
        time_seq, batch_size, *_ = xs.shape

        hs = (hs[0].reshape(batch_size, self.cell_size),
              hs[1].reshape(batch_size, self.cell_size))

        xs = torch.relu(self.input_layer(xs))

        hiddens = []
        for x, mask in zip(xs, h_masks):
            hs = (hs[0] * (1 - mask), hs[1] * (1 - mask))
            hs = self.cell(x, hs)
            hiddens.append(hs[0])
        hiddens = torch.cat([h.unsqueeze(0) for h in hiddens], dim=0)
        outs = self.output_layer(hiddens)

        return outs, hs


class QNetLSTM(nn.Module):
    def __init__(self, observation_space, action_space, h_size=1024, cell_size=512):
        super(QNetLSTM, self).__init__()
        self.h_size = h_size
        self.cell_size = cell_size
        self.rnn = True

        self.input_layer = nn.Linear(
            observation_space.shape[0] + action_space.shape[0], self.h_size)
        self.cell = nn.LSTMCell(self.h_size, hidden_size=self.cell_size)
        self.output_layer = nn.Linear(self.cell_size, 1)

        self.output_layer.apply(mini_weight_init)

    def init_hs(self, batch_size=1):
        new_hs = (next(self.parameters()).new(batch_size, self.cell_size).zero_(), next(
            self.parameters()).new(batch_size, self.cell_size).zero_())
        return new_hs

    def forward(self, ob, ac, hs, h_masks):
        time_seq, batch_size, *_ = ob.shape

        hs = (hs[0].reshape(batch_size, self.cell_size),
              hs[1].reshape(batch_size, self.cell_size))

        xs = torch.cat([ob, ac], dim=-1)
        xs = torch.relu(self.input_layer(xs))

        hiddens = []
        for x, mask in zip(xs, h_masks):
            hs = (hs[0] * (1 - mask), hs[1] * (1 - mask))
            hs = self.cell(x, hs)
            hiddens.append(hs[0])
        hiddens = torch.cat([h.unsqueeze(0) for h in hiddens], dim=0)
        outs = self.output_layer(hiddens)

        return outs, hs


class ModelNetLSTM(nn.Module):
    def __init__(self, observation_space, action_space, h_size=200, cell_size=200):
        super(ModelNetLSTM, self).__init__()
        self.h_size = h_size
        self.cell_size = cell_size
        self.rnn = True

        self.input_layer = nn.Linear(
            observation_space.shape[0] + action_space.shape[0], self.h_size)
        self.cell = nn.LSTMCell(self.h_size, hidden_size=self.cell_size)
        self.output_layer = nn.Linear(
            self.cell_size, observation_space.shape[0])
        self.output_layer.apply(weight_init)

    def init_hs(self, batch_size=1):
        new_hs = (next(self.parameters()).new(batch_size, self.cell_size).zero_(), next(
            self.parameters()).new(batch_size, self.cell_size).zero_())
        return new_hs

    def forward(self, ob, ac, hs, h_masks):
        time_seq, batch_size, *_ = ob.shape

        hs = (hs[0].reshape(batch_size, self.cell_size),
              hs[1].reshape(batch_size, self.cell_size))

        xs = torch.cat([ob, ac], dim=-1)
        xs = torch.relu(self.input_layer(xs))

        hiddens = []
        for x, mask in zip(xs, h_masks):
            hs = (hs[0] * (1 - mask), hs[1] * (1 - mask))
            hs = self.cell(x, hs)
            hiddens.append(hs[0])
        hiddens = torch.cat([h.unsqueeze(0) for h in hiddens], dim=0)
        outs = self.output_layer(hiddens)

        return outs, hs


class DiscrimNet(nn.Module):
    def __init__(self, observation_space, action_space, h1=32, h2=32):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(
            observation_space.shape[0] + action_space.shape[0], h1)
        self.fc2 = nn.Linear(h1, h2)
        self.output_layer = nn.Linear(h2, 1)
        self.apply(weight_init)

    def forward(self, ob, ac):
        h = torch.tanh(self.fc1(torch.cat([ob, ac], dim=1)))
        h = torch.tanh(self.fc2(h))
        return self.output_layer(h)


class DiaynDiscrimNet(nn.Module):
    def __init__(self, f_space, skill_space, h_size=300, discrim_f=lambda x: x,):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(f_space.shape[0], h_size)
        self.output_layer = nn.Linear(h_size, skill_space.shape[0])
        self.apply(weight_init)
        self.discrim_f = discrim_f

    def forward(self, ob):
        feat = self.discrim_f(ob)
        h = torch.relu(self.fc1(feat))
        return self.output_layer(h)
