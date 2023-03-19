import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
from tqdm import tqdm


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    """PPO算法,采用截断方式"""

    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        actor_lr,
        critic_lr,
        lmbda,
        epochs,
        eps,
        gamma,
        device,
    ):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):  # 传入的是一个字典，包含了一条序列的数据
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(
            self.device
        )
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float
        ).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        td_target = rewards + self.gamma * self.critic(next_states) * (
            1 - dones
        )  # 计算td_target
        # 在给出的代码中，1 - dones 表示下一个状态是否为终止状态的标志位。
        # 在强化学习中，智能体与环境交互的过程通常被建模成一个马尔可夫决策过程（MDP），
        # 其中智能体通过观察当前状态并执行一个动作来转移到下一个状态，并获得一个即时奖励。如果下一个状态是终止状态，
        # 那么智能体无法再从该状态获得任何奖励或执行任何动作。在这种情况下，我们通常将 TD 目标设置为当前状态的即时奖励，而不考虑下一个状态的价值。
        # 在这里，1 - dones 的作用是将终止状态的标志位取反，使其在计算 TD 目标时起到判断下一个状态是否为终止状态的作用。
        # 具体来说，如果下一个状态是终止状态，那么 dones 将被设置为 1，1 - dones 将变为 0，此时计算 TD 目标时只考虑当前状态的即时奖励。
        # 否则，如果下一个状态不是终止状态，那么 dones 将为 0，1 - dones 将变为 1，此时计算 TD 目标时将考虑下一个状态的价值估计。

        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(
            self.gamma, self.lmbda, td_delta.cpu()
        ).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            # 具体来说，ratio 张量表示当前策略下执行某个动作的概率，与旧策略下执行该动作的概率之比。在 PPO 算法中，
            # 使用这个比率来评估新策略相对于旧策略的改进程度，从而确定优化策略的方向和程度。
            # 然而，如果比率的变化过大，可能会导致算法不稳定，因此需要对其进行截断操作。
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
            # torch.clamp 函数接受三个参数，分别为输入张量、下限值和上限值，它会将输入张量中小于下限值的元素替换成下限值，
            # 大于上限值的元素替换成上限值，其余元素保持不变。在这里，torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
            # 将 ratio 张量中的元素限制在区间 [1 - eps, 1 + eps] 内，其中 eps 是 PPO 算法中的一个超参数，表示截断范围的大小。
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach())
            )
            # 在这里，td_target.detach() 的作用是将 td_target 张量与计算图分离，即将其从后向传播的计算中剥离，使其成为一个常量。
            # 这样可以避免在优化策略时对 td_target 张量的梯度进行更新，从而保证优化的正确性和稳定性。
            # 通常情况下，当需要对一个张量进行截断或分离操作时，我们可以使用 detach() 函数来实现。
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


if __name__ == "__main__":

    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = "CartPole-v0"
    env = gym.make(env_name)
    env.seed(0)  #
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(
        state_dim,
        hidden_dim,
        action_dim,
        actor_lr,
        critic_lr,
        lmbda,
        epochs,
        eps,
        gamma,
        device,
    )

    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

    torch.save(agent.actor.state_dict(), "ppo_cartpole.pth")  # 保存模型
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("PPO on {}".format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 21)
    plt.plot(episodes_list, mv_return)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("PPO on {}".format(env_name))
    plt.show()
~