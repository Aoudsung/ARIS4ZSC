# R015 原始回报确定界推导：test_time_simple 的 [G_min, G_max]

**日期：** 2026-07-13
**性质：** 静态推导（只读网络访问 + 仓库检索），无任何本地或远端运行。
**对应冻结槽位：** [PATH_C_OPPORTUNITY_AUDIT_SPEC.md](PATH_C_OPPORTUNITY_AUDIT_SPEC.md) §6"原始回报界"。
**结论先行：** 确定界为 `[G_min, G_max] = [−400, +400]`。代入 §3.3 冻结的样本量规则得每原型
259,850 个配对块（四原型合计约 104 万块、≥312 万个正式臂回合、约 1.25×10⁹ 环境步，尚未计
规划分叉）。**§3.3 的"当前设计精度不可行"条款在推导阶段即触发**：按纯 Hoeffding 确定界
规则，R015 在该基底上不可执行；正式运行前必须重新设计区间规则（§7 列出选项，属 Type-B
决策）。这是设计判定，不是科学读数。

---

## 1. 来源绑定

- 项目锁定版本：`jaxmarl==0.1.0`（`pyproject.toml:22`；注释于 `requirements.txt:8`）。
- 本推导读取 GitHub tag `v0.1.0` 下四个源文件（只读获取）：
  - `jaxmarl/environments/overcooked_v2/settings.py`（奖励与时间常数）
  - `jaxmarl/environments/overcooked_v2/overcooked.py`（奖励发生点与交互前条件）
  - `jaxmarl/environments/overcooked_v2/layouts.py`（`test_time_simple`/`test_time_wide` 网格）
  - `jaxmarl/environments/overcooked_v2/common.py`（静态物件枚举）
- **冻结义务：** 冻结时必须对远端虚拟环境实际安装的上述四文件计算 SHA-256 并绑定；若与
  tag `v0.1.0` 内容不一致，本推导作废重做。

## 2. 原始回报的奖励源枚举（逐处）

`overcooked.py` 中基础奖励累加器 `reward` 的全部写入点恰有两处；塑形累加器
`shaped_reward` 另有四处，走独立通道：

1. **交付（唯一双向奖励源）：**
   `reward += successful_delivery * select(is_correct_recipe, 1, negative_rewards ? −1 : 0) * DELIVERY_REWARD`，
   其中 `DELIVERY_REWARD = 20`（`settings.py`），前条件
   `successful_delivery = object_is_goal ∧ inventory_is_dish`。官方 Test Time 配置
   `negative_rewards=true`（`configs/ocv2_test_time_simple_standard.yaml:7`），故正确交付
   +20、错误交付 −20。
2. **按钮式配方指示器激活：**
   `reward −= successful_indicator_activation * INDICATOR_ACTIVATION_COST`（成本 5），前条件
   要求 `BUTTON_RECIPE_INDICATOR`（网格字符 `L`）。**两个 Test Time 网格都只含被动指示器
   `R`、没有任何 `L`**（`test_time_simple` 网格 `WW2WWWWW / W  WB  0 / R AWPA X / W  WB  1 /
   WW2WWWWW`；`test_time_wide` 末行 `WWRWWW`），故该项在两布局恒为零。
3. **塑形四项**（进锅 3、开火 5、取菜 5、取盘 3，负奖励分支含 −3）：写入 `shaped_reward`
   独立累加器，不进入基础 `reward`。本项目标准评估只汇 `raw_episode_return`
   （`path_c_standard_evaluation.py` 全文无 shaped 引用），R020 配置亦注明塑形只在训练期
   加权（`path_c_standard_formal_simple.yaml:32-34`）。
4. 无每步时间惩罚、无终止事件奖励；`URGENCY_CUTOFF=40`、`INDICATOR_ACTIVATION_TIME=10`
   是观测/动力学常数，无奖励效应。

因此原始回合回报 = 20 × (正确交付数) − 20 × (错误交付数)。

## 3. 交付的消耗性（封死刷分路径）

交互后库存更新为
`new_inventory = successful_pickup * (pile_ingredient + merged_ingredients) + no_effect * inventory`，
且 `no_effect = ¬successful_pickup ∧ ¬successful_drop ∧ ¬successful_delivery`。交付成立时
两项系数均为零，故 `new_inventory = 0`（空手）——**无论配方正确与否，交付都消耗手中的
成品盘**。又因 `inventory_is_dish` 只能通过从锅中取出煮熟的汤获得，任何一次交付（正负两向）
都消耗一份完成烹饪的汤。错误交付不可原地重复刷分。

## 4. 锅吞吐上界

`test_time_simple` 网格恰有 **1 个锅**。每份可交付的汤必须在该锅完成一段
`POT_COOK_TIME = 20` 步的连续烹饪（计时器每环境步减一；开火由第三份食材触发或自动开火，
烹饪时长与食材构成无关）；同一锅的烹饪区间两两不相交。回合长 400 步，故完成烹饪的汤数
≤ ⌊400/20⌋ = 20，从而

`D_total = 正确交付数 + 错误交付数 ≤ 20`。

（用首次装填与末次搬运的网格几何还能收紧到 ≤19，对结论量级无影响，不采用；上界只用
"烹饪区间不相交"这一条不可辩驳的机制事实。）

## 5. 确定界与冻结规则代入

- `G_max = +20 × 20 = +400`（20 份全部正确交付）；
- `G_min = −20 × 20 = −400`（20 份全部错误交付）；
- 配对差确定宽度 `W_D = 2 × (G_max − G_min) = 1600`；
- `R_delivery = 20 ⇒ m = 0.25 × 20 = 5`。

代入 §3.3 冻结规则（`α_e = 0.0125`，`K = 4`）：

`n = ⌈2 × 1600² × ln(2/0.0125) / (4 × 5²)⌉ = ⌈2 × 2,560,000 × 5.07517 / 100⌉ = 259,850 / 原型`。

四原型合计 1,039,400 个配对块；每块三个正式臂 × 400 步 ⇒ 仅正式回合约 **1.25×10⁹ 环境步**，
规划分叉再高 1–2 个数量级。对照：本审计要保护的十单元训练预算总计 3×10⁸ 环境步。

**符号形式（供后续布局复用）：** 当 `G_min = −G_max`（对称负奖励）且 `m = α·R_delivery` 时，

`n = 32 · D_max² · ln(2/α_e) / (K · α²)`，即 α=0.25、K=4 下 `n ≈ 650 · D_max²`。

交叉验证：代码测试锚定的 `n = 650` 恰是 `D_max = 1`（±20 玩具界）的取值——公式实现与
本推导一致，玩具界与真实界的差距是 `D_max²` = 400 倍。

- `test_time_wide`：2 个锅 ⇒ `D_max ≤ 40` ⇒ `n ≈ 650 × 1600 ≈ 1.04×10⁶ / 原型`，更不可行。
- 调大 `m` 救不了：`n ∝ 1/α²`，把每原型 n 压到 2,000 需 `α ≈ 2.85`（要求每局净收益 2.85 次
  正确交付 ≈ 57 原始分，而已发表 Fictitious Co-Play 的 cross-play 均值只有 6）——实际意义
  门槛先于统计变得荒谬。

## 6. 判定

按 §3.3 原文："若由真实回报界得到的 `n` 在预算内不可执行，必须在正式运行前记录'当前设计
精度不可行'并重新设计审计；不得改用试点方差缩小确认性样本量。"本推导即该记录。触发原因
是纯 Hoeffding 规则对确定宽度的平方依赖（`n ∝ W_D²`）撞上"400 步内锅吞吐允许 ±20 次交付"
的机制事实，与实现质量无关。

### 6.1 机械锁步条件下的配对差范围

未触发探查的配对块中，三个正式实验组使用同一冻结规划器、相同的被动屏蔽信念更新和相同
随机数。只要逐步重放确认完整轨迹逐字节相同，三组原始回报相同，故

`Delta_net=Delta_response=Delta_cost=0`。

触发探查的配对块把 `t_p` 定义为已经完成 `t_p` 个环境步后的咨询边界。首次触发满足
`1<=t_p<=100`；嵌入轨迹记录编号为 0 到 399，共同前缀覆盖记录 0 到 `t_p-1`，分支从记录
`t_p` 开始，因此触发后最多剩 `400-t_p<=399` 步。逐步重放还必须确认三个实验组在 `t_p` 前的完整轨迹前缀相同；
于是前缀奖励在任一配对差中精确抵消。

但是，不能由 `399` 步直接推出后缀至多交付 `floor(399/20)=19` 份汤。§4 的不相交烹饪区间
论证界定的是完整回合：任意截取的后缀可能继承一份在 `t_p` 前已经开始烹饪的汤，也可能继承
已经完成但尚未交付的汤。这样的交付发生在后缀内，却不需要把完整 20 步烹饪区间包含在后缀
内。当前登记的锁步检查没有加入“触发时不存在烹饪中、已完成或暂存的汤”这一额外条件。

因此，仅凭已登记且可机械检查的事实，每个实验组的触发后回报仍只能用完整回合界
`[-400,400]` 约束，任一配对差满足

`|D| <= 400-(-400) = 800`。

这个界只在共同前缀的逐字节锁步检查通过时使用；检查失败则整个配对块作废。800 也是当前
已登记信息能保证的最小统一常数：这些检查没有排除共同前缀留下不同正确性的已完成任务对象，
随后两个分支分别选择交付不同对象并产生相反符号的后续交付；因此它们没有排除后缀回报接近
+400 与 −400 的配对。要证明更小常数，必须新增并机械检查触发状态的在制品、库存和暂存物
约束，不能把它当作现有锅吞吐证明的推论。

## 7. 已采用的按轮效应区间与触发次数证书

项目负责人已用本节设计替代 §3.3 的不可执行 Hoeffding 规则。试点仍是每原型 20 个配对块，
只查接线，不进入正式数据，也不决定正式轮数。

1. **正式轮次。** 每轮对四个原型各生成一个配对块，轮向量按同一冻结规则独立生成；主体固定为
   `agent_1` 厨师席，伙伴原型固定为 `agent_0` 信号席，不得按运行后计数或结果筛选、替换或重排轮次。三个正式实验组的
   后续动作都经注册延续控制器 `map_prototype_committed_cook_v1` 产生；该具体形式不改变本推导的配对差范围。端点的轮均值
   `D_bar_r=(1/4)sum_k D_{r,k}` 保持四原型等权，且位于 `[-800,800]`。正式上限为
   `N_rounds_max=2500`，共 10,000 个配对块；三个正式实验组各走 400 步，约为
   `10000*3*400=12,000,000` 个环境步，规划分叉另报。
2. **一次效应区间。** 只在 2500 轮且此前未按触发次数停止时，对 `Delta_net` 和
   `Delta_response` 各计算一次 Maurer–Pontil empirical Bernstein 区间。每项
   `alpha_EB=0.010`，范围宽度 `b-a=1600`，半宽为
   `sqrt(2*V_N*ln(2/alpha_EB)/N) + 7*1600*ln(2/alpha_EB)/(3*(N-1))`。
   区间截到 `[-800,800]`。共同随机数配对产生的正式轮均值样本方差 `V_N` 决定可达到的
   宽度；即使 `V_N=0`，2500 轮时第二项仍约为 7.9153，不能省略。
3. **只读触发次数的提前停止。** checkpoint 冻结为轮数 `{200,400,800,1600,2500}`。
   触发次数证书族 `alpha_rho=0.005` 等分为每次 `alpha_rho_j=0.001`。零触发时，令总块数
   `B=N*K`。抽样清单必须机械确认四个原型的触发前缀分别使用独立随机流，触发前没有轮级共享
   随机源；只有在这个条件下才有
   `P(零触发)=prod_k(1-rho_k)^N<=exp(-N*K*rho_bar)`，从而
   `rho_bar_U=ln(1000)/B`。`800*rho_bar_U<5` 等价于
   `B>800*ln(1000)/5=1105.2408...`。块数必须是 4 的倍数，所以纯算术上最少为 1108 块
   （277 轮）；冻结 checkpoint 中首次能停止的是 400 轮，即 **1600 块**。200 轮时上界给出
   `800*rho_bar_U=ln(1000)=6.9078...>5`，尚不能停止；400 轮时为
   `ln(1000)/2=3.4539...<5`。
4. **非零触发。** 每个原型使用单侧精确 Clopper–Pearson 上界，单个原型错误率
   `alpha_rho_j/K=0.00025`，四个上界等权平均。只有 `800*rho_bar_U<m` 才提前判定注册
   控制器达不到门槛。该输入只含触发次数，不含回报或任何配对差；若未停止，效应值只在最终
   2500 轮查看一次。若提前停止，后续 checkpoint 和正式回合不再生成；已完成块的轨迹保留，
   但提前停止器不读取其中的回报或配对差，也不组装最终效应区间。
5. **总体保证。** 安全证书族 0.025、`Delta_net` 区间 0.010、`Delta_response` 区间
   0.010、触发次数证书族 0.005，相加为 0.05；并集上界给出至少 0.95 的联合保证。

“约 805 块即可在零触发时停止”只会在同时使用原建议的 760 常数、并且错误地不把 0.005
分到五次 checkpoint 时出现。按最终的 800 界和登记的 Bonferroni 分配，正确数字是理论最少
1108 块、冻结日程实际最早 1600 块。

## 8. 本推导引用的原始出处

- `https://raw.githubusercontent.com/FLAIROx/JaxMARL/v0.1.0/jaxmarl/environments/overcooked_v2/settings.py`
- `https://raw.githubusercontent.com/FLAIROx/JaxMARL/v0.1.0/jaxmarl/environments/overcooked_v2/overcooked.py`
- `https://raw.githubusercontent.com/FLAIROx/JaxMARL/v0.1.0/jaxmarl/environments/overcooked_v2/layouts.py`
- `https://raw.githubusercontent.com/FLAIROx/JaxMARL/v0.1.0/jaxmarl/environments/overcooked_v2/common.py`
- 仓库内：`configs/ocv2_test_time_simple_standard.yaml`、`path_c_standard_formal_simple.yaml`、
  `path_c_standard_evaluation.py`、`pyproject.toml`。
