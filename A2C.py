import torch
from torch import nn, optim
from torch.distributions import Categorical
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64, weights=[1.0, 0.5, 2.0], entropy_decay=0.98):
        super(ActorCritic, self).__init__()

        # TODO:1.隐层设计；2.critc和actor是否共用前几层；3.优化器调整
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size*2),
            nn.Tanh(),
            nn.Linear(hidden_size*2, hidden_size*2),
            nn.Tanh(),
            nn.Linear(hidden_size*2, action_dim),
            nn.Softmax(dim=1),
        )

        # self.optimizer = optim.SGD(self.parameters(), lr=0.001)
        self.optimizer = optim.Adam(self.parameters())
        self.weights = weights
        self.entropy_decay = entropy_decay
        self.state_buffer = []
        self.prob_buffer = []
        self.value_buffer = []
        self.action_buffer = []

    def forward(self, states):
        values = self.critic(states)
        probs = self.actor(states)

        return probs, values

    def choose_action(self, states, buffer=True):
        probs, values = self.forward(states)
        # print("values:", values)
        # print("probs:", probs)
        actions = Categorical(probs).sample()

        if buffer:
            self.state_buffer.append(states)
            self.value_buffer.append(values)
            self.prob_buffer.append(probs)
            self.action_buffer.append(torch.unsqueeze(actions, 1))
        # print("actions:", actions)
        actions = actions.cpu().numpy() + 1
        values = values.detach().cpu().numpy()
        probs = probs.detach().cpu().numpy()
        return actions, values, probs

    def train_model(self, rewards, next_state):
        states = torch.cat([next_state])
        values = torch.cat(self.value_buffer)
        actions = torch.cat(self.action_buffer)
        distributions = Categorical(torch.cat(self.prob_buffer))

        # print(states)
        # print("values:", values)
        # print("actions:", actions)
        # print("probs:", self.prob_buffer)
        # print(distributions.logits)

        returns = self.compute_returns(states, rewards)
        # print("returns:", returns)
        advantage = returns - values
        # print(advantage)
        critic_loss = advantage.pow(2)
        # print(distributions.log_prob(actions.squeeze()).unsqueeze(1))
        actor_loss = -distributions.log_prob(actions.squeeze()).unsqueeze(1) * advantage.detach()
        entropy = torch.unsqueeze(distributions.entropy(), 1)

        # print("actor_loss:", actor_loss)
        # print("critic_loss:", critic_loss)
        # print("entropy_loss:", entropy)
        if self.weights[2] > 0.1:
            self.weights[2] = self.weights[2] * self.entropy_decay
        total_loss = (self.weights[0]*critic_loss + self.weights[1]*actor_loss - self.weights[2]*entropy).mean()
        # print("total_loss:", total_loss)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.state_buffer = []
        self.prob_buffer = []
        self.value_buffer = []
        self.action_buffer = []

        return actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy(), entropy.detach().cpu().numpy(), total_loss.detach().cpu().numpy()

    def compute_returns(self, next_state, rewards, gamma=0.99):
        value = self.critic(next_state)
        returns = []
        # print("value:", value)
        for step in reversed(range(len(rewards))):
            value = rewards[step] + gamma * value
            returns.insert(0, value)
        # print("compute_returns:", rewards, torch.cat(returns))
        # returns = torch.Tensor(returns)
        # print("returns:", returns)
        # returns = returns.unsqueeze(1)
        return torch.cat(returns)
    
    def get_buffer(self):
        return self.state_buffer, self.prob_buffer, self.value_buffer, self.action_buffer
    
    def load_buffer(self, state_buffer, prob_buffer, value_buffer, action_buffer):
        self.state_buffer, self.prob_buffer, self.value_buffer, self.action_buffer = \
            state_buffer, prob_buffer, value_buffer, action_buffer
    
    def get_optimizer(self):
        return self.optimizer.state_dict()
    
    def load_optimizer(self, checkpoint):
        self.optimizer.load_state_dict(checkpoint)
