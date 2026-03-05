# Dialog 流程：需要改/写的文件

参考学长 traver 的用法：**make_prompt 生成 role_desc → 建 Player → Arena + Environment 跑对话 → 存 jsonl**。

---

## 1. 做 prompt 的模块（新建或补全）

**作用**：根据「一条 profile + 一条 corpus」生成 User 和 Assistant 的 `role_desc`。

- **可选位置**：`interaction/make_prompt.py`，或在 `user_utils.py` / `assistant_utils.py` 里写函数。
- **输入**：
  - profile 行：`translated_user_profile.jsonl` 的一行（至少要有 `player`）
  - corpus 行：`emotion_appraisal_corpus.tsv` 的一行（`Sentence` + 7 个 appraisal 列）
- **输出**：
  - `user_desc`：把 `{persona}`、`{situation}`、`{appraisal_values}` 填进 `interaction/prompt/user_base.txt`
  - `assistant_desc`：若用 `assistant_base.txt` 就填好占位符；否则可以是一段固定说明（如「你是支持性 NPC，回复用户」）

**traver 对照**：`traver/utils/make_prompt.py` 里的 `prompt_student` / `prompt_tutor`，按一条数据生成一个 role_desc。

---

## 2. agent_user.py（interaction/chatarena/）

- **现状**：和基类 `Player` 一样，用 `name`、`role_desc`、`backend` 即可。
- **建议**：**不用改**。主流程里直接 `Player(name="User", role_desc=user_desc, backend=user_backend)`，`user_desc` 由上面的 make_prompt 生成（已含 persona + situation + appraisal）。
- 只有在你希望「User 每轮用不同 system prompt」时，才需要写一个子类在 `act()` 里动态拼 role_desc；一般 1 轮对话不需要。

---

## 3. agent_assistant.py（interaction/chatarena/）

- **现状**：带 KT、verifier、多回复选优等，是 tutoring 用的。
- **建议**：**二选一**  
  - **A**：主流程里不用 `Assistant`，改用 `Player(name="Assistant", role_desc=assistant_desc, backend=assistant_backend)`，和 User 对称。  
  - **B**：保留 `Assistant` 但删掉 KT/verifier 相关，只做「根据 observation 调 backend.query 返回一句回复」。

---

## 4. 主流程：interaction/run_dialog.py（你要实现的 run_base）

- **作用**：实现「读数据 → 配对 → 建 Agent → 跑对话 → 存结果」。
- **步骤**：
  1. 读 `translated_user_profile.jsonl`、`emotion_appraisal_corpus.tsv`；按需随机或按 id 配对 (profile, corpus_row)。
  2. 对每一对：用 **make_prompt** 得到 `user_desc`、`assistant_desc`。
  3. 建 User、Assistant（见上），建 `Conversation(player_names=["User", "Assistant"])`，`Arena(players=[User, Assistant], environment=env)`。
  4. 跑 1 轮或 N 轮（如 `arena.launch_cli(max_steps=2)` 或自己写循环 `arena.step()`），把 `env.get_observation()` 存成 jsonl。
  5. 命令行参数：profile 路径、corpus 路径、user/assistant 的 model 或 vllm endpoint、输出路径、是否 limit 条数等。

**traver 对照**：`traver/run_base.py` 的 `interactive_simulation`（遍历 prompt_data → 建 tutor/student/moderator → Arena + launch_cli → 写 jsonl）。

---

## 5. scripts/run_dialog.sh

- **作用**：调主流程脚本，传参。
- **内容**：设置环境变量或参数，例如：
  - `--profile_file`, `--corpus_file`, `--output_file`
  - `--user_model`, `--assistant_model` 或 `--vllm_endpoint_user`, `--vllm_endpoint_assistant`
  - `--max_rounds`（如 1）、`--limit`
- 最后一行：`python interaction/run_dialog.py ...`

---

## 小结表

| 文件 | 要做的事 |
|------|----------|
| **make_prompt（或 user_utils/assistant_utils）** | 写「profile + corpus 行 → user_desc, assistant_desc」 |
| **agent_user.py** | 可不改，用 `Player` + make_prompt 的 user_desc |
| **agent_assistant.py** | 要么主流程用 `Player` 当 Assistant，要么简化 Assistant 去掉 KT/verifier |
| **run_dialog.py** | 主流程：读数据、配对、make_prompt、建 Arena、跑对话、写 jsonl |
| **run_dialog.sh** | 调用 `run_dialog.py` 并传参 |

先实现 **make_prompt** 和 **run_dialog.py**，再补 **run_dialog.sh**，agent 用现有 `Player` 即可跑通。
