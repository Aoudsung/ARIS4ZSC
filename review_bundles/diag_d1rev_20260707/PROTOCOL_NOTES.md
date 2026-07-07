# D1-rev 执行过程记录(diag_d1rev_20260707)

**预注册**:D1 预注册 §9(2026-07-07,先于任何 rev 数据冻结)。commit `da32140`。
**授权**:用户"ok执行"(D1-rev 全链,签 C 直通,分支表绑定)。

## 评审链(thread 019f386c)

- 部署前:BLOCK(any-recipe 门、空掩码崩溃)→ 修复 → **APPROVE-WITH-NITS**
  (nit:空 K 划分的 NaN 使方向一致性保守化,照录)。
- 签 C 交叉评审:九项逐项 CONFIRM / CONFIRM-WITH-CAVEATS,总判
  **CONFIRM-WITH-CAVEATS**;caveats 全部采纳(见 EXPERIMENT_LOG 条目)。

## golden 人工核验(PASS)

- D1 首轮饱和的状态被新门正确排除:ego 持汤(dp011)、partner 持汤(dp019–021)
  的决策点 gate_main=0;
- 取汤发起从原始库存位转移正确编码(dp010 ego 装盘、dp018 partner 取汤);
- 窗口边界无差一(dp005/dp006 分界);持盘不关门(dp017 后 gate 保持)符合 §9;
- 类分布即刻健康:52 门内点上 18/20/14 三向分布。

## 充分性门(一轮 PASS,零软失败)

indist_val 16922(最小类 3129)、blind_terminal 27947(4458)、dev 11185、
blind_offaxis 13641、indist_train 67725。歧义剔除率与截尾率照录于 gate 报告。

## 终读要点(单看已消耗;完整数字见 readout_frozen.json)

- R1 FAIL:G_indist=+2.5e-4(CI [+5.0e-5,+4.9e-4]——统计可检出、量级微观);
- R2 技术性 TRUE 但**空洞**(门槛=0.5×2.5e-4),照录不作证据;K 方向一致;R3 翻正;
- 状态充分性:NOHIST 分布内 0.9986 / 盲 0.9979;严格门(就绪+空手)双侧 1.0000;
  ID-oracle 0.9988 与 NOHIST 平齐(仅分布内参照);
- 阳性对照:blind-ingredient-near-neutral 历史增益 **+0.125**(仪器有效的存在性证明);
- dev 负增益 -0.0296(过下限,advisory)——OOD 历史负资产第二次独立复现。

## 签 C 裁决(用户预授权直通)

分支表字面:两轮均落第 1 行。跨仪器实质:**基底在终端轴上未实例化"隐藏倾向推断"
现象**(在公共状态+有效选项+ego 选项条件化之后无可辨残差倾向,且状态充分性迁移至盲伙伴)。
predictive 预训练与 population 多样化两条转向在本基底上均不被支持;承载下一步的决策 =
基底/伙伴重设计(新的 Type-B 用户决策,超出 D1 范围)。
D1/D1-rev 仪器沉淀为任何新基底的**现象存在性证书**。
