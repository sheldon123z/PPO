## TRPO理解

在TRPO算法中，$r_t(\theta)$ 表示代理在状态 $s_t$ 选择动作 $a_t$ 的概率比率，即使用当前策略 $\pi_\theta(a_t|s_t)$ 选择动作 $a_t$ 的概率与使用旧策略 $\pi_{\theta_{old}}(a_t|s_t)$ 选择动作 $a_t$ 的概率之比，即：

$
r_t(\theta)=\frac{\pi_\theta\left(a_t \mid s_t\right)}{\pi_{\theta_{\text {old }}}\left(a_t \mid s_t\right)}
$

如果 $r_t(\theta)>1$，则说明当前策略比旧策略更好；如果 $r_t(\theta)<1$，则说明当前策略更差。

TRPO算法的优化目标是最大化一个“替代”目标函数（surrogate objective function），该函数的表达式为：
$
L^{s u r}(\theta)=\mathbb{E}_t\left[r_t(\theta) A_t\right]
$
其中 $\mathbb{E}_t$ 表示对时间步 $t$ 的期望，$A_t$ 表示优势函数的估计值，用于衡量在状态 $s_t$ 采取动作 $a_t$ 相对于采取平均动作的价值提升程度。优势函数的估计值通常使用 GAE (Generalized Advantage Estimation) 方法计算得到。

TRPO算法最大化替代目标函数的同时，需要满足一个约束条件，即当前策略 $\pi_\theta(a_t|s_t)$ 与旧策略 $\pi_{\theta_{old}}(a_t|s_t)$ 的 KL 散度不超过一个指定的阈值。通过限制策略更新的幅度，TRPO算法可以确保策略改进的稳定性和效率。





## 优势函数的表达式

在 PPO 算法中，优势函数（Advantage Function）用于衡量在当前状态下采取某个动作相对于平均情况的价值提升，即一个动作相对于其他动作的优劣程度。

优势函数的表达式通常是：

$A_t = Q_t - V_t$

其中 $Q_t$ 表示在时间步 $t$ 采取某个动作后能够获得的总回报，$V_t$ 表示在时间步 $t$ 采取随机策略时期望能够获得的总回报。$A_t$ 表示在时间步 $t$ 采取某个动作相对于随机策略的价值提升。

在实际应用中，$Q_t$ 和 $V_t$ 都无法直接获取，需要通过近似方法进行估计。在 PPO 算法中，$Q_t$ 和 $V_t$ 可以通过价值网络来估计，$V_t$ 可以直接通过价值网络输出的值来获取，而 $Q_t$ 则需要进一步通过优势函数来计算。具体来说，$Q_t$ 的表达式可以写成：

$Q_t = r_t + \gamma V_{t+1}$

其中 $r_t$ 表示在时间步 $t$ 采取某个动作后获得的即时奖励，$\gamma$ 表示折扣因子，$V_{t+1}$ 表示在时间步 $t+1$ 时采取随机策略的期望回报。将 $Q_t$ 的表达式代入优势函数的定义式中，可以得到：

$A_t = Q_t - V_t = r_t + \gamma V_{t+1} - V_t$

这个表达式就是在 PPO 算法中通常所采用的优势函数的表达式。在具体实现中，可以使用 PyTorch 提供的张量运算函数来计算优势函数的值，例如 `torch.sub()` 函数可以用于计算两个张量的差值，`torch.mul()` 函数可以用于计算两个张量的乘积。


## PPO 算法通常使用神经网络来表示策略和价值函数。其算法逻辑主要包括以下步骤：

1. 定义神经网络表示策略 $\pi_{\theta}(a|s)$ 和价值函数 $V_{\phi}(s)$，其中 $\theta$ 和 $\phi$ 分别是策略网络和价值网络的参数。

2. 对于每个时间步 $t$，通过神经网络得到当前状态 $s_t$ 的动作 $a_t$ 和对应的概率 $\pi_{\theta}(a_t|s_t)$。

3. 执行动作 $a_t$，得到下一个状态 $s_{t+1}$ 和对应的奖励 $r_t$。

4. 计算当前策略的优势函数 $A_t$，即 $A_t = \sum_{i=t}^T \gamma^{i-t} r_i - V_{\phi}(s_t)$，其中 $T$ 是当前回合的最大步数，$\gamma$ 是折扣因子。

5. 计算当前策略和旧策略之间的概率比值 $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$，以及对应的 clipped 概率比值 $r_t^c(\theta) = clip(r_t(\theta), 1-\epsilon, 1+\epsilon)$，其中 $\epsilon$ 是一个小的正数。

6. 根据 clipped 概率比值和优势函数计算 surrogate objective，即 $L_t^{clip}(\theta) = min(r_t^c(\theta)A_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)$。

7. 使用梯度下降方法更新策略和价值网络的参数，即 $\theta \leftarrow \theta + \alpha \nabla_{\theta} L_t^{clip}(\theta)$ 和 $\phi \leftarrow \phi + \alpha \nabla_{\phi} L_t^{VF}(\phi)$，其中 $\alpha$ 是学习率。

8. 重复执行步骤 2 到步骤 7，直到达到设定的最大回合数或达到收敛条件。


## PPO-clip为什么取cliped objective 和unclipped objective的 lower bound
PPO-clip算法是基于优化目标函数的思想，目标函数包含了两个部分：一个是策略评估的loss，另一个是正则项。

在PPO-clip算法中，对于策略评估的loss，我们需要找到一个平衡点，既要保证策略更新方向的正确性，又要保证更新后的策略与之前的策略不要相差太大，以避免更新过度。因此，我们需要用到一个clipped objective的技巧，将策略更新的大小限制在一个合理的范围内。

具体来说，PPO-clip算法采用的是PPO算法的一种改进，通过引入一个clipping参数，限制了新策略和旧策略之间的KL散度，从而确保策略更新不会太大，而同时保证更新方向的正确性。当新策略的KL散度大于clipping参数时，我们取clipped objective和unclipped objective的lower bound，即两者的最小值作为策略更新的loss，这样就可以在保证更新方向正确的同时，限制更新大小，防止过度更新。当KL散度小于clipping参数时，我们可以直接采用unclipped objective作为策略更新的loss。

因此，PPO-clip算法之所以要取clipped objective和unclipped objective的lower bound，是为了确保策略更新的正确性和大小适度。