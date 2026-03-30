# 两阶段生成：改动清单

## 一、整体流程变化

| 原来 | 现在 |
|------|------|
| 每行：1 次 seed_event + 7 次单维度 expansion | 每行：1 次 seed_event + **1 次 Step1（所有 0/3 维度一起）** + **1 次 Step2（所有 1/2 维度一起）** |
| 每维单独生成，彼此无约束 | Step1 先出一批「骨架句」→ Step2 在完整 scenario 上只做局部补充 |

---

## 二、Prompt 改动

### 2.1 现有单维度 prompt（Attention / Certainty / …）  
**结论：可以保留不动，但不再被主流程调用。**

- 当前是「event + 一个维度 + 定义/例子 → 一句 expansion」，适合单维生成。
- 新流程改成「一批维度一起生成」，所以需要**新的批量 prompt**，单维 prompt 可留作参考或以后单句补写用。

### 2.2 需要新增的两个 prompt

#### （1）Step 1：Anchor 批量生成 — 新文件 `prompt/step1_anchor_batch.txt`

**输入（由代码拼好传入）：**
- `{EVENT}`：种子事件（seed_event）
- `{EMOTION}`：情绪（小写）
- `{ANCHOR_LIST}`：所有取值为 0 或 3 的维度，每项包含：
  - 维度英文名（如 Attention, Certainty）
  - 取值 0 或 3
  - 该维度的**定义**（和现有 Attention.txt 等里的 Definition 一致）
  - **当前状态描述**（用 seed2scenario.py 里的 `*_state(value)` 得到的那句）

**任务说明（建议写进 prompt）：**
- 根据上述 event 与情绪，为**每一个**列出的 anchor 维度各写**一句** expansion。
- 这些句子合起来要像**同一个情境**的多个侧面：时间、人物、事实一致，不互相矛盾。
- 每句只负责把「这一维度的 appraisal 状态」用情境细节体现出来，不要重复定义、不要写「我觉得……」这种内心独白。

**输出格式（便于解析）：**  
例如规定模型严格按下面格式输出，方便用正则或按行解析：

```
Attention:
<一句 expansion>

Certainty:
<一句 expansion>
…
```

或统一用「维度名: 一句」也可以，只要和解析代码约定好。

**微调建议：**
- 在 prompt 里写清：**一次生成多句、且多句必须同属一个情境**，这是和以前单维生成的最大区别。
- 若某行没有任何 0/3 维度（全是 1/2），则 Step1 不调用，代码里用「空 anchor 列表」分支处理（见下）。

---

#### （2）Step 2：Softer 批量生成 — 新文件 `prompt/step2_softer_batch.txt`

**输入：**
- `{EXPANDED_SCENARIO}`：Step1 生成的全部 anchor 句子拼成的一段（或按维度标好的多句），即「当前已确定的 scenario 正文」。
- `{EMOTION}`：情绪（小写）
- `{SOFTER_LIST}`：所有取值为 1 或 2 的维度，每项包含：
  - 维度英文名
  - 取值 1 或 2
  - 该维度的**定义**
  - **当前状态描述**（同样用 `*_state(value)`）

**任务说明：**
- 上面这段 expanded scenario 已经固定，**不能改、不能否定**。
- 请仅为列出的每个 softer 维度各写**一句** expansion，且必须：
  - 与已有 scenario 一致（同一情境、同一批事实），
  - 只做**局部补充**（补充细节、侧面），不重写已有内容。

**输出格式：**  
与 Step1 约定同一种格式，例如：

```
Effort:
<一句 expansion>

Pleasantness:
<一句 expansion>
…
```

**微调建议：**
- 强调「仅允许在已有 scenario 上做局部补充」和「不得与已有句子矛盾」。
- 若某行没有任何 1/2 维度（全是 0/3），则 Step2 不调用，只保留 Step1 的句子。

---

## 三、代码 / 逻辑改动（run_seed2scenario.py）

### 3.1 不再调用的部分
- 不再对 7 个维度逐个调用 `get_appraisal_expansion(..., prompts[dim], ...)`。
- 可以保留 `get_appraisal_expansion` 和单维 `prompts` 的加载逻辑，以备后用；主流程改为两步批量。

### 3.2 新增/修改的逻辑

1. **划分 anchor / softer**
   - 对当前行，根据 7 个维度的取值把维度名分成两组：
     - `anchor_dims`：取值 in (0, 3)
     - `softer_dims`：取值 in (1, 2)

2. **拼 Step1 的 `{ANCHOR_LIST}`**
   - 对每个 `anchor_dims` 中的维度：
     - 用现有 `state_fns[dim](row[col_map[dim]])` 得到 state 文本；
     - 从「维度名 → 定义」的映射里取定义（可写死在代码里，或从各 Attention.txt 等里抽一行 Definition）。
   - 拼成一段带编号或小标题的文本，填入 prompt 的 `{ANCHOR_LIST}`。

3. **调用 Step1**
   - 若 `anchor_dims` 非空：用 `step1_anchor_batch.txt` + `EVENT` / `EMOTION` / `ANCHOR_LIST` 调一次 API；解析返回得到 `expansions[dim]` for dim in anchor_dims。
   - 若 `anchor_dims` 为空：不调 Step1，`expanded_scenario` 可设为 `seed_event` 或空字符串（见下）。

4. **拼 Step2 的 `{EXPANDED_SCENARIO}`**
   - 若 Step1 被调用过：把 Step1 返回的所有 anchor 句子按固定顺序拼成一段（或「维度名: 句子」逐行），作为 `EXPANDED_SCENARIO`。
   - 若 Step1 未调用（没有 0/3）：可用 `seed_event` 作为唯一已有内容，或显式写「当前仅有事件描述：{seed_event}」。

5. **拼 Step2 的 `{SOFTER_LIST}`**
   - 与 Step1 的 ANCHOR_LIST 类似，为每个 1/2 维度拼「维度名 + 取值 + 定义 + state 描述」。

6. **调用 Step2**
   - 若 `softer_dims` 非空：用 `step2_softer_batch.txt` + `EXPANDED_SCENARIO` / `EMOTION` / `SOFTER_LIST` 调一次 API；解析返回得到 `expansions[dim]` for dim in softer_dims。
   - 若 `softer_dims` 为空：不调 Step2。

7. **合并并写 JSON**
   - `expansions` = Step1 的 anchor 结果 ∪ Step2 的 softer 结果（若某维既不是 0/3 也不是 1/2 不会出现，因为取值只有 0–3）。
   - 仍用现有 `build_scenario(row, seed_event, expansions, scenario_id)` 得到最终 JSON，写入 JSONL。

### 3.3 解析 Step1 / Step2 的返回

- 约定模型严格按「维度名: 或 维度名:\n」后面跟一句 expansion，用正则或按块 split 解析出 `{ "attention": "...", "certainty": "...", ... }`。
- 维度名到小写 key 的映射要一致（attention, certainty, effort, pleasantness, responsibility, control, circumstance），注意 TSV 里是 Control / Pleasant，对应 control / pleasantness。

### 3.4 边界情况

| 情况 | 处理 |
|------|------|
| 某行 7 个维度全是 0/3 | 只跑 Step1，不跑 Step2；Step2 的 expansion 为空，但该行不会有 1/2 维度，所以 7 句都来自 Step1。 |
| 某行 7 个维度全是 1/2 | 只跑 Step2；Step1 不跑，`EXPANDED_SCENARIO` 用 `seed_event` 或「事件：{seed_event}」作为已有内容。 |
| 解析失败（某维度缺句或格式乱） | 可重试该步一次，或记 log 并跳过该行 / 用占位句，看你要的鲁棒性。 |

---

## 四、小结：你需要做的具体事

1. **Prompt**
   - 新增 `prompt/step1_anchor_batch.txt`：event + emotion + 所有 0/3 维度（含定义与 state）→ 多句同一情境的 expansion，并约定输出格式。
   - 新增 `prompt/step2_softer_batch.txt`：已生成的 expanded scenario + emotion + 所有 1/2 维度（含定义与 state）→ 仅做局部补充的 expansion，并约定输出格式。
   - 现有单维 prompt 可保留不删，主流程不再用。

2. **代码**
   - 按行划分 anchor_dims / softer_dims，拼 ANCHOR_LIST / SOFTER_LIST（含定义与 state）。
   - 实现 Step1 调用与解析 → 得到 anchor 的 expansions，并拼成 EXPANDED_SCENARIO。
   - 实现 Step2 调用与解析 → 得到 softer 的 expansions。
   - 合并 expansions，照旧 build_scenario + 写 JSONL；处理「全 anchor」「全 softer」两种边界。

3. **定义从哪里来**
   - 要么在代码里维护一个 `DIM_DEFINITIONS = {"attention": "...", "certainty": "...", ...}`（从现有 prompt 里抄一行 Definition）；
   - 要么在 prompt 目录下放一个 `dimension_definitions.txt` 或按维度的小文件，代码里读入再拼进 ANCHOR_LIST / SOFTER_LIST。

这样改完后，流程就是：**seed_event → Step1（0/3 一批）→ Step2（在 Step1 完整内容上补 1/2）→ 每维一句的 scenario JSON**，且 prompt 通过「定义 + 状态描述 + 任务说明」把 appraisal 含义和两阶段约束都写清楚。
