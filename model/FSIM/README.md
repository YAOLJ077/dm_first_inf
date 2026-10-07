# 无性孢子囊起始的初季侵染路线

本模型表示研究假设：部分亚热带地区在目标果园首个病斑出现前，附近二季果果园或其他活体病部可能持续提供无性孢子囊。这里的“常年存在”指区域菌源反复补充，不是同一批孢子囊全年存活。目标果园的首次病斑不是模型启动条件；目标站气象代替周边源区气象也是一项必要的空间近似。

过程为：**潜在源区夜间产孢 → 附着存活 → 飞散/沉降 → 脱离存活 → 游动孢子释放/存活 → 叶片感染 → 潜育 → 显症窗口**。每个有正产孢量的夜晚形成独立队列。当前版本只重建这些源区队列导致的侵染，不将模拟的新病斑再反馈为新的产孢源，不融合孢子捕获观测。

## 依据与必要假设

| 过程 | 本版本实现 | 来源或性质 |
| --- | --- | --- |
| 夜间产孢 | RH ≥ `sporulation_rh_min_pct` 或降水 > 0；夜间湿润 ≥ `minimum_moist_hours`，10 < 平均温度 < 30°C | Brischetto 2021 pp.3–5；当前配置为98.539%和6.533 h，可手动调节 |
| 产孢相对量 | Lalancette Richards 曲面，除以 20°C、24 h 的值，限制到 0–1 | Brischetto Fig.2A 的归一化方法；数值系数沿用原 FSIM |
| 附着/脱离死亡 | x=T(1−RH/100)，每小时累加 1/[24(c0−c1x+c2x²)]，剩余量为初始量×max(0,1−累计死亡) | 原 FSIM 的 Brischetto/Blaeser–Weltzien 系数；附着 (9.27,1.12,0.04)，脱离 (5.67,0.47,0.02) |
| 飞散和沉降 | 达到 `wind_min_m_s` 时按 D&DR=1 进入脱离态飞散队列；无雨时保持飞散，雨量达到 `effective_deposition_min_rain_mm` 且低于 `heavy_rain_washoff_mm` 时有效沉降，达到大雨阈值则冲刷/脱落 | INRAE 葡萄霜霉病生物学页面指出孢子囊主要由风传播；Brischetto 2021 p.4 的合并抽象；雨量阈值为本研究的显式敏感性假设 |
| 叶面湿润 | DPD = 气温 − 露点；降雨优先，DPD < 2.0°C 湿润，DPD > 3.8°C 干燥，中间保持上一小时状态 | 本路径内部的因果天气代理；与死亡经验指标 x、源区产孢湿润规则分开 |
| 游动孢子释放 | WD ≥ exp[(-1.022+19.634)/TWD]，TWD>0 | Rossi 2008 p.484 印刷公式，与当前 oospore 一致；借用到无性孢子囊是跨路线假设 |
| 游动孢子存活 | 释放后的经过小时/湿润小时 > 1 即永久死亡 | Rossi Eq.7；本实现相当于释放后首次干燥即终止 |
| 感染 | 释放后逐小时更新当前 `WD×TWD` 快照，再乘当前物候敏感系数 `S`；首次达到 `infection_degree_hours` 即感染 | Rossi (2008) p.485 的逐小时 `WD×TWD` 判断加入当前阶段调节；当前S路径阈值240是本研究的显式适配参数，不是Rossi原文的60 |
| 潜育和显症 | 两条固定温度响应累加至 1，给出显症起止窗口 | Rossi Eqs.8–9；包含感染当小时 |

产孢响应、死亡、释放、潜育方程与系数集中在 [calculation.py](calculation.py)；夜间产孢开关及两个雨量阈值由配置覆盖。附着、飞散和沉降后的脱离状态不重复计算同一小时的死亡；脱离死亡从风速触发飞散时刻开始，贯穿飞散和沉降阶段，并在沉降后的存活期内继续计算。无雨不产生背景沉降，强降雨在有效沉降前终止该队列。沉降后仍须在脱离态存活期内完成游动孢子释放，释放后转入游动孢子存活，感染后转入潜育。

Rossi PDF 的释放指数为 `(-1.022+19.634)/TWD`；旧 FSIM 用过 `-1.022+19.634/TWD`，两者不同，本版本使用前者。Brischetto 原文的感染/严重度模块并未照搬，本版本按研究设计使用 Rossi 后段，所以这是有明确假设的组合路线，不是两篇论文任一模型的完整原样复刻。

保留的计算约定为北京时间前日 20:00 至当日 07:59 统计夜间、08:00 建立队列。它不是当地天文夜长。S 从 2 月 1 日 00:00 开始计算，紧接 P 在 1 月 31 日全天结束的评估窗口；O 仍从 1 月 1 日 01:00 开始。这是本研究的计算约定，不是实测生物学启动日期。S 首夜只累计 2 月 1 日 00:00–07:00 的 8 个小时，排除 1 月 31 日晚间，标记为不完整夜间；满足产孢条件后最早于 2 月 1 日 08:00 建立队列。没有初始任意剂量注入、病斑年龄衰减或按模拟显症日期重置菌源。未实现 Brischetto 原文按母病斑反复产孢次数的衰减，因为本假设对应未知的区域多源病斑，而非已知单个母病斑。

参数值和研究假设集中记录在本目录的 `config.json` 与 `calculation.py`；固定系数表示沿用的过程关系，不意味着已完成当地生物学验证。当前 site-year 没有实测叶面湿度，使用输入露点计算 DPD 代理；若旧天气行没有露点，则由气温和相对湿度反推。

## 配置与调用

[config.json](config.json) 中显式保留感染阈值和风速传播门槛等需要人工测试的参数。DPD 叶面湿润关系及其数值阈值固定写在 `calculation.py` 中，不作为待调参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `sporulation_parameters.sporulation_rh_min_pct` | 75.0% | 夜间产孢判定的最低 RH；可人工修改 |
| `sporulation_parameters.minimum_moist_hours` | 5.0 h | 夜间产孢判定所需的最短湿润时间；可人工修改 |
| `deposition_parameters.effective_deposition_min_rain_mm` | 3.122 mm/h | 有效沉降所需的最低小时雨量；低于此值不产生背景沉降 |
| `deposition_parameters.heavy_rain_washoff_mm` | 9.588 mm/h | 大雨冲刷/脱落阈值；达到或超过此值时队列不形成有效沉降 |
| `biological_parameters.infection_degree_hours` | 139.821°C·h | 释放后 `WD×TWD` 感染阈值，可人工修改；这是S路径适配参数 |
| `dispersal_mode` | `wind_gated` | 逐小时风速门控的默认传播方式；`merged` 仅用于结构对照 |
| `structure_parameters.wind_min_m_s` | 2.039 m/s | 允许孢子囊从源区转移的最低风速，可人工修改；不是文献拟合系数 |

可复制配置为另一个 JSON，修改产孢阈值、雨量阈值、感染阈值或风速门槛，再通过 `--config` 指定。没有自动调参过程。默认 `dispersal_mode=wind_gated`，`structure_parameters.wind_min_m_s=2.039`。`merged` 可用于不使用风速门控的结构对照。Brischetto 源区夜间产孢仍使用 RH、降雨和可用 `leaf_wetness_minutes` 判据，不能用 DPD 代理替代。

```bash
python run/run_sporangia.py \
  --weather data/weather_data/site_year/23_83_108_1_2022.json \
  --config model/FSIM/config.json \
  --output-dir result/FSIM/23_83_108_1_2022
```

输入仍为露天栽培、北京时间的 site-year JSON。模型不设置独立的寄主存在门控；感染是否达到阈值由湿润、温度、游动孢子存活和物候敏感系数共同决定。

## 输出

每个 site-year 输出一个 JSON，`manifest.json` 汇总首个感染及首个预测显症窗口。`process.cohorts` 保存每个产孢队列及其失败或感染状态；`infections` 保存感染事件和潜育/显症窗口。`cohort_id`/`event_id` 可追踪同一队列。不同队列同小时感染仍分别保留，同时提供 `unique_infection_hours`。

`detached_survival_end_reason` 区分死亡、释放、冲刷和天气结束；`status` 还会标记 `airborne_censored`、`sporangium_dead_before_deposition` 和 `sporangium_washed_off_before_deposition`。天气不足时保留感染事件并标记 `incubation_censored`；不会以年末代替显症日期。相对剂量用于追踪队列存活，不乘到感染阈值中，也不输出未经标定的感染概率。schema_version=3 不接受旧版仅到沉降的配置。
