from tqdm import tqdm
import numpy as np
import torch
import collections
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (
        cumulative_sum[window_size:] - cumulative_sum[:-window_size]
    ) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[: window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, num_episodes):
    """
    这是一个用于训练 PPO 算法的函数。该函数使用给定的智能体（agent）在给定的环境（env）上进行一定数量（num_episodes）的训练，并返回一个列表，
    其中包含每一轮训练所得到的回报值。具体地，该函数通过多次循环来进行训练。
    在每一轮循环中，首先重置环境并获取当前状态，然后利用智能体的策略网络（agent.take_action(state)）
    来选择一个动作，并执行该动作，获取下一个状态、即时奖励和终止信号。接着，将当前状态、动作、下一个状态、即时奖励和终止信号存储在一个字典中（transition_dict），以便在后续的更新过程中使用。
    在每个轮次结束时，将该轮次的回报值添加到 return_list 列表中，并调用智能体的 update 方法来更新策略网络和价值网络的参数。
    具体来说，该方法首先使用字典中存储的数据来计算出优势函数的值，并根据优化目标来计算出策略网络和价值网络的损失函数，并使用反向传播算法来更新网络的参数。
    为了方便用户了解训练的进度和效果，该函数还使用了 tqdm 模块来显示训练的进度条和回报值统计信息。每完成一定数量的轮次后，该函数将打印出该轮次的平均回报值，并更新进度条的显示信息。
    总之，该函数实现了一个简单的 PPO 算法的训练流程，可以通过适当调整函数的参数来进行定制化的训练过程。
    """
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {
                    "states": [],
                    "actions": [],
                    "next_states": [],
                    "rewards": [],
                    "dones": [],
                }
                state = env.reset(seed=0)
                done = False
                while not done:
                    # env.render()
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict["states"].append(state)
                    transition_dict["actions"].append(action)
                    transition_dict["next_states"].append(next_state)
                    transition_dict["rewards"].append(reward)
                    transition_dict["dones"].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "episode": "%d" % (num_episodes / 10 * i + i_episode + 1),
                            "return": "%.3f" % np.mean(return_list[-10:]),
                        }
                    )
                pbar.update(1)
    return return_list


def train_off_policy_agent(
    env, agent, num_episodes, replay_buffer, minimal_size, batch_size
):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            "states": b_s,
                            "actions": b_a,
                            "next_states": b_ns,
                            "rewards": b_r,
                            "dones": b_d,
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "episode": "%d" % (num_episodes / 10 * i + i_episode + 1),
                            "return": "%.3f" % np.mean(return_list[-10:]),
                        }
                    )
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
