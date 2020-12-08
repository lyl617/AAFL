import torch
from A2C import ActorCritic

device = torch.device("cuda")
kwargs = {'num_workers': 1, 'pin_memory': True}

agent = ActorCritic(4, 10).to(device)

states = [1,1,1,1]
actions, values, probs = agent.choose_action(torch.Tensor(states).unsqueeze(0).to(device))

print(actions, values, probs)

