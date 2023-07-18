import torch

class A2C(torch.nn.Module):
    def __init__(self,input_shape, layer1, kernel_size1, stride1, layer2, kernel_size2, stride2, layer3, kernel_size3, stride3, fc1_dim, out_actor_dim, out_critic_dim):
        super(A2C, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=input_shape, out_channels=layer1, kernel_size=kernel_size1, stride=stride1)
        self.conv2 = torch.nn.Conv2d(in_channels=layer1, out_channels=layer2, kernel_size=kernel_size2, stride=stride2)
        self.conv3 = torch.nn.Conv2d(in_channels=layer2, out_channels=layer3, kernel_size=kernel_size3, stride=stride3)
        
    
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(in_features=64*7*7, out_features=fc1_dim)
        self.out_actor = torch.nn.Linear(in_features=fc1_dim, out_features=out_actor_dim)
        self.out_critic = torch.nn.Linear(in_features=fc1_dim, out_features=out_critic_dim)

    
    def forward(self,x, hidden=None):
        fx = x.float() / 255.
        
        out_backbone = self.conv1(fx)
        out_backbone = self.relu(out_backbone)
        out_backbone = self.conv2(out_backbone)
        out_backbone = self.relu(out_backbone)
        out_backbone = self.conv3(out_backbone)
        out_backbone = self.relu(out_backbone)
        flat = self.flatten(out_backbone)
        
        out_linear = self.fc1(flat)
        out_linear = self.relu(out_linear)
        #actor
        actor = self.out_actor(out_linear)
        #critic
        critic = self.out_critic(out_linear)
        return actor,critic