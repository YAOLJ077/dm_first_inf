
'''
The primary inoculum source persists long-term as a "mycelial pool (inoculum source pool) in litter/twigs/fallen fruits" and can produce spores at night;
Nocturnal wetness triggers sporulation after meeting the threshold, with sporulation intensity determined by MP/TMP;
Daytime mortality is driven by VPD, with no mortality at night (related to sunlight);
Dispersal events are divided into rainfall-type and gust-type, with rainfall-type being the dominant factor when both occur simultaneously;
A single dispersal event fully transfers the attached-state reservoir as a cohort (complete transfer hypothesis);
The detached state does not increase further, experiencing mortality only during daytime until it falls below the threshold;
Wetness events encountered during the detached-state survival period trigger ZRE;
Zoospore survival is bound to the end of the wetness period (SUZ), and infection must occur within the survival period;
Symptom expression is temperature-driven, with accumulated incubation creating a window period
'''


'''
S0: Inoculum Sources in the Area
S1: Night-time Sporulation from Inoculum Sources
S2: Attached Sporangia Inventory Dynamics
S3: DIS ----- Dispersal Event, S3a. Rain-driven dispersal, S3b. Wind-driven dispersal
S4: Dispersed Sporangia Inventory Dynamics
S5: DEP ----- Deposition Event
S6: Deposited Sporangia Inventory Dynamics
S7: ZRE ----- Zoospore Release Event
S8: SUZ ----- Zoospore Survival Period
S9: INF ----- Infection Event
S10: Incubation -----Oil Spot
'''


'''
For each sporangia cohort:

  ┌─[A] sporangia alive window (from dispersal → sporangia_alive_end)
  │
  │   ┌─[B] deposition must occur inside [A]
  │   │
  │   │   ┌─[C] zoospore release
  │   │   │   - only inside [A]
  │   │   │   - requires leaf wetness
  │   │   │   - requires WD ≥ WDmin(TWD)
  │   │   │
  │   │   │   ┌─[D] zoospore survival window (SUZ)
  │   │   │   │   - start = release_datetime
  │   │   │   │   - end   = first dry leaf
  │   │   │   │
  │   │   │   │   ┌─[E] infection
  │   │   │   │   │   - only inside SUZ
  │   │   │   │   │   - WD x TWD ≥ threshold
  │   │   │   │   │
  │   │   │   │   └─ success → incubation

'''

'''
时间 ─────────────────────────────────────────────────────────────────▶

[Attached pool on leaf]
        │
        │  夜间湿润 + Tmin < T < Tmax
        ▼
┌──────────────────────┐
│ Sporulation (S1)     │  ← 累积 attached_pool
└──────────────────────┘
        │
        │  rain ≥ dispersal.rain_min
        │  或 wind ≥ dispersal.wind_min
        ▼
┌──────────────────────┐
│ Dispersal event      │  ← dispersal_datetime
│ (rain / wind)        │
└──────────────────────┘
        │
        │  S4:脱离态孢子囊存活开始
        │  （连续死亡,VPD 驱动）
        ▼
┌────────────────────────────────────────────┐
│ Sporangia survival window (S4)              │
│ D(t+1)=D(t)*exp(-λ(VPD))                     │
│                                              │
│ 有效窗口：                                  │
│ [dispersal_datetime , sporangia_alive_end]  │
└────────────────────────────────────────────┘
        │
        │  若是 wind-dispersal:
        │  需要 rain ≥ deposition.rain_min
        ▼
┌──────────────────────┐
│ Deposition           │  ← deposition_datetime
│ (rain gate)          │
└──────────────────────┘
        │
        │  ⚠ 注意：释放只能在 sporangia 仍活着时发生
        ▼
┌────────────────────────────────────────────┐
│ Zoospore release window (S5)                 │
│ 条件：                                      │
│  - leaf wet (rain 或 VPD ≤ thr)              │
│  - WD ≥ exp((-1.022+19.634)/TWD)              │
│                                              │
│ 释放时间： zoospore_release_datetime         │
└────────────────────────────────────────────┘
        │
        │  释放瞬间 → 游动孢子诞生
        ▼
┌────────────────────────────────────────────┐
│ Zoospore survival window (SUZ, S6)           │
│ 条件：                                      │
│  - leaf wet (rain 或 VPD ≤ thr)              │
│                                              │
│ 有效窗口：                                  │
│ [release_datetime , suz_end]                 │
│ suz_end = 第一次“叶片变干”的时刻            │
└────────────────────────────────────────────┘
        │
        │  ⚠ infection 必须发生在 SUZ 内
        ▼
┌────────────────────────────────────────────┐
│ Infection window (S7)                        │
│ 条件：                                      │
│  - leaf wet                                 │
│  - WD x TWD ≥ infection_condition           │
│                                              │
│ infection_datetime                          │
└────────────────────────────────────────────┘
        │
        │
        ▼
┌──────────────────────┐
│ Incubation           │
│ (thermal accumulation)│
└──────────────────────┘
        │
        ▼
 oil_spot_start  ──►  oil_spot_end

'''