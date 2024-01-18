import torch
import torch.nn as nn
import torch.optim as optim

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(input_dims[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(torch.prod(torch.tensor(dims.size())))

    def forward(self, state):
        conv1 = torch.relu(self.conv1(state))
        conv2 = torch.relu(self.conv2(conv1))
        conv3 = torch.relu(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0], -1)
        flat1 = torch.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)

        return actions
