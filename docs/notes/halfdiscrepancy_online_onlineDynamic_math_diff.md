# halfdiscrepancy、online、onlineDynamic 的数学更新差别

## 1. 先说结论

这三个名字里，真正对应“更新机制”的其实是：

- `halfdiscrepancy` 对应基线半差异方法，默认 `delta_update_mode="refit"`。
- `online` 对应 `delta_update_mode="online"`。
- `onlineDynamic` 对应 `delta_update_mode="online_dynamic"`。

所以更准确的比较方式是：

- `halfdiscrepancy` = 半差异基线 + `refit`
- `halfdiscrepancy-online` = 半差异路径 + `online`
- `halfdiscrepancy-onlineDynamic` = 半差异路径 + `online_dynamic`

三者共同的观测模型都可以写成

`y_t(x) = rho * eta(x, theta_t) + delta_t(x) + eps_t,   eps_t ~ N(0, sigma_eps^2)`.

共同点是：

- PF 仍然追踪 `theta_t`。
- half-discrepancy 的含义仍然是“PF 权重侧不直接吃 discrepancy，BOCPD / expert 预测侧使用 discrepancy”。
- 真正不同的地方，是 discrepancy 的残差标签如何构造、旧历史会不会被重新解释、以及状态是静态 GP 还是动态滤波状态。

## 2. 方法名和代码语义的对应关系

- `R-BOCPD-PF-halfdiscrepancy`
  - `use_discrepancy=False, bocpd_use_discrepancy=True`
  - 默认 `delta_update_mode="refit"`
- `R-BOCPD-PF-halfdiscrepancy-online`
  - 同样是 half-discrepancy 路径
  - 但把 `delta_update_mode` 改成 `"online"`
- `R-BOCPD-PF-halfdiscrepancy-onlineDynamic`
  - 同样是 half-discrepancy 路径
  - 但把 `delta_update_mode` 改成 `"online_dynamic"`

因此，`halfdiscrepancy` 不是与 `online`、`onlineDynamic` 完全同层级的三个并列对象；严格说，后两者是前者这条 discrepancy 路径上的两种新更新规则。

## 3. 模式一：halfdiscrepancy 基线，其实就是 refit

### 3.1 数学对象

在共享 discrepancy 的 half-discrepancy 基线里，expert 历史中的每个观测点 `s <= t`，都会用“当前时刻 t 的 PF 后验”重新计算残差：

`r_s^(refit,t) = y_s - rho * sum_i w_(t,i) * eta(x_s, theta_(t,i))`.

然后用整段保留下来的 expert 历史

`D_t^delta = {(x_s, r_s^(refit,t)) : s in retained history}`

重新拟合 discrepancy GP。

### 3.2 数学含义

这个更新不是严格 online 的。因为当 PF 粒子位置或权重在时刻 `t` 变化后，连旧样本 `s < t` 的 discrepancy 标签都会被改写。

所以它的本质是：

- discrepancy 不是“过去冻结下来的误差记忆”；
- discrepancy 是“附着在当前 PF 后验上的一个重新解释后的预测律”；
- 旧历史会随着新的 PF 后验被反复重解释。

### 3.3 直观理解

如果把 discrepancy 看作“模拟器之外剩下的那一部分”，那么在 `refit` 里，这个“剩下的部分”不是一次记录后永久保存，而是每来一个新 batch，就用最新的 `theta` 后验重新分账一次。

所以：

- 历史会被重标注；
- 新数据会通过改变 PF 后验，间接影响所有旧残差；
- 当前 discrepancy 是一个“全历史、全量重算”的对象。

## 4. 模式二：online = 冻结残差 + append-only 静态 GP

### 4.1 残差怎么形成

在 `online` 下，新到一批数据时，只在该批到来当下形成一次冻结残差：

`tilde(r)_t = y_t - rho * sum_i w_(t,i) * eta(x_t, theta_(t,i))`.

这个残差一旦写入 discrepancy 数据集，之后就不再因为未来 PF 状态变化而被改写。

### 4.2 更新规则

于是 discrepancy 历史变成 append-only：

`D_t^delta = D_(t-1)^delta U {(x_t, tilde(r)_t)}`.

随后用固定超参数的静态 GP 条件在全部已追加残差上：

`delta(x) | D_t^delta ~ GP posterior on appended frozen residuals`.

这里关键点有两个：

- 旧残差冻结，不重解释；
- latent discrepancy 本身仍然是“静态函数”的 GP 后验，只是条件数据越来越多。

### 4.3 和 refit 的核心差别

`online` 与 `refit` 的根本差别不在于“有没有历史”，而在于“旧历史的标签是否会被今天的 PF 后验重写”。

- `refit`：会重写。
- `online`：不会重写。

因此，`online` 的历史语义更接近真实在线系统中的日志积累：

- 当时怎么算出来的残差，就永久保留成当时的标签；
- 后来 `theta_t` 再变，也不能回头改账。

### 4.4 状态复杂度和记忆特征

虽然叫 online，但这里的 discrepancy 仍是静态 exact GP 记忆：

- 状态规模会随着追加的残差点数增长；
- 没有显式的时间传播方程；
- 新数据的影响是“多加一个条件点”，不是“先做状态预测，再做测量更新”。

所以它更像：

- frozen residual memory
- static GP with growing dataset

而不是动态状态空间模型。

## 5. 模式三：onlineDynamic = 冻结残差 + 动态潜变量滤波

### 5.1 关键变化不在残差，而在 discrepancy 状态本身

`onlineDynamic` 里，新 batch 的共享残差同样在到达时冻结：

`r_t = y_t - rho * sum_i w_(t,i) * eta(x_t, theta_(t,i))`.

所以它和 `online` 一样，都不再回头重写旧残差标签。

真正不同的是，`onlineDynamic` 不把 discrepancy 当成“一个静态 GP 函数，条件在越来越多的数据上”，而是把它当成“随时间漂移的动态潜状态”。

### 5.2 状态参数化

代码里用一个小的 RBF basis 展开：

`delta_t(x) = phi(x)^T beta_t`.

这里 `beta_t` 是需要随时间递推的潜在系数状态。

### 5.3 先传播，再更新

在 `onlineDynamic` 中，每个 batch 的数学动作是两步：

第一步，先把上一时刻后验传播成当前先验：

`beta_t | D_(1:t-1) ~ N(beta_(t-1), P_(t-1) / lambda + q I)`.

其中：

- `lambda` 是 forgetting 因子；
- `q I` 是过程噪声；
- 这一步表示 discrepancy 自己会漂移，且旧信息会逐渐变旧。

第二步，再用当前 batch 的冻结残差做线性高斯更新：

`r_t = Phi_t beta_t + noise`.

然后用 Kalman / recursive least-squares 形式把 `(beta_t, P_t)` 更新到新后验。

### 5.4 和 online 的核心差别

`online` 的状态逻辑是：

- 旧残差固定；
- GP 本身不做时间传播；
- 新数据只是往静态函数后验里再加条件信息。

`onlineDynamic` 的状态逻辑是：

- 旧残差也固定；
- 但 discrepancy 后验会先扩散 / 遗忘 / 漂移，再吃下新数据；
- 因而“新数据更重要”不是口头解释，而是体现在 `P_(t-1) / lambda + q I` 这个递推里。

所以它比 `online` 多出来的数学假设是：

- discrepancy 不是固定函数；
- discrepancy 是时变潜状态；
- 时间递推本身就是模型的一部分。

### 5.5 为什么它不是“online 的小改版”

从计算图上说，`online` 和 `onlineDynamic` 的差别不是“同一个 GP 多了个 buffer”，而是：

- `online` 维护的是 growing GP memory；
- `onlineDynamic` 维护的是 fixed-dimensional latent state `(beta_t, P_t)`。

因此 `onlineDynamic` 的状态维度由 basis 数决定，而不是由历史长度决定。

## 6. 三种模式放在一起看

### 6.1 残差标签是否会被重写

- `halfdiscrepancy / refit`
  - 会。旧历史残差使用当前 PF 后验重新计算。
- `online`
  - 不会。每个 batch 到来时冻结一次残差后永久保留。
- `onlineDynamic`
  - 不会。残差同样冻结。

### 6.2 discrepancy 被当成什么数学对象

- `refit`
  - 当前 PF 后验下、基于保留历史重新拟合的静态 GP 预测律。
- `online`
  - 冻结残差驱动的静态 GP 在线记忆。
- `onlineDynamic`
  - 冻结残差驱动的动态潜状态滤波器。

### 6.3 新数据如何影响旧信息

- `refit`
  - 通过改变当前 PF 后验，回头改写整段历史的 discrepancy 标签。
- `online`
  - 不改写旧标签，只把新标签追加到静态 GP 数据集中。
- `onlineDynamic`
  - 不改写旧标签，但会通过遗忘因子和过程噪声，降低远历史对当前状态的约束。

### 6.4 状态维度如何变化

- `refit`
  - 每次重拟合都依赖保留历史长度。
- `online`
  - GP 条件集随历史增长，状态规模随数据增长。
- `onlineDynamic`
  - 主要状态是固定维度的 `(beta_t, P_t)`，只与 basis 维数有关。

### 6.5 对“时间漂移”的建模态度

- `refit`
  - 没有显式漂移状态，更多是“当前 PF 后验下的重解释”。
- `online`
  - 也没有显式漂移状态，只是把残差按时间顺序积累。
- `onlineDynamic`
  - 显式假设 discrepancy 会漂移，并用递推方程描述这种漂移。

## 7. 如果只用一句话概括

- `halfdiscrepancy / refit`：每一轮都用“当前 theta 后验”重新解释整个历史，所以旧 discrepancy 标签会变。
- `online`：旧标签冻结不变，但 discrepancy 还是静态 GP，只是数据集不断追加。
- `onlineDynamic`：旧标签同样冻结，但 discrepancy 本身被建模成时变隐状态，先传播再更新，因此新数据天然更重。

## 8. 和 refresh / STDGate 的关系

这三种模式在 standardized-gate refresh 下也不同：

- `refit`
  - refresh 后直接基于保留下来的 recent window 重新做一次 refit。
- `online`
  - refresh 会截断最近窗口，只保留 recent frozen residual buffer，然后重建静态 online GP。
- `onlineDynamic`
  - refresh 会截断最近窗口，然后仅用 recent buffer 重建动态 basis filter。

所以：

- 对 `refit` 来说，refresh 更像“缩短后重新全量解释”；
- 对 `online` 来说，refresh 更像“砍掉旧记忆，再继续 append-only”；
- 对 `onlineDynamic` 来说，refresh 更像“做一次硬重置，但后续每步仍按动态滤波递推”。

## 9. 一个更实用的选择标准

如果你关心的是：

- “当前 PF 后验下，历史应该被重新解释”
  - 选 `halfdiscrepancy / refit`
- “历史标签一旦形成就不该回改，但 discrepancy 还是静态函数”
  - 选 `online`
- “历史标签不回改，而且 discrepancy 本身会随时间漂移，新数据应该天然更重要”
  - 选 `onlineDynamic`

## 10. 本文依据的仓库实现

本文说明是根据下列实现和 workdoc 整理的：

- `rolled_cusum_modeling_workdoc.md`
- `calib/restart_bocpd_rolled_cusum_260324_gpytorch.py`
- `calib/particle_specific_discrepancy.py`
- `calib/configs.py`

其中最关键的代码语义是：

- `delta_update_mode="refit"`：旧历史 discrepancy 标签会按当前 PF 后验重建；
- `delta_update_mode="online"`：冻结残差、append-only、静态 GP；
- `delta_update_mode="online_dynamic"`：冻结残差、递推传播、动态 basis filter。
