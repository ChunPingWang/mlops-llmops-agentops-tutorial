# ADR-001：LLMOps / MLOps / AgentOps 工具鏈 PoC 驗證計劃

> **ADR 編號**: ADR-001  
> **狀態**: 提議（Proposed）  
> **日期**: 2026-06-01  
> **決策者**: Application Architect  
> **硬體環境**: 通用 x86 工作站 / 本地 GPU 推論 + 雲端 LLM Fallback

---

## 第一部分：Architecture Decision Record (ADR)

### 1. 背景（Context）

團隊正在評估建立企業級 AI 維運平台的可行性，涵蓋 LLMOps（大型語言模型運維）、MLOps（機器學習運維）與 AgentOps（自主代理運維）三大領域。在正式投入生產環境之前，需透過 PoC 驗證工具鏈的整合性、成熟度與適用性。

先前已完成 vLLM + LiteLLM 架構設計（見 vLLM-Deployment-Architecture-Plan.md），本 ADR 聚焦於以 Docker Compose 部署全棧工具鏈，搭配本地 GPU 推論（Gemma 4 26B）作為主要後端，雲端 LLM API 作為 Fallback。

### 2. 決策（Decision）

**採用「本地 GPU 推論 + 雲端 LLM Fallback」架構，以 Docker Compose 一站式部署以下工具鏈進行 PoC 驗證：**

| 層級 | 工具 | 角色 |
|------|------|------|
| AI Gateway | LiteLLM Proxy | 多模型路由、Fallback、成本追蹤 |
| LLM 可觀測性 | Langfuse | Trace、Token 用量、品質評估 |
| 向量資料庫 | Qdrant | RAG 語意檢索 |
| 安全護欄 | NeMo Guardrails | PII 過濾、輸入/輸出安全（雙模式：應用層攔截 + LiteLLM Hook） |
| Agent 框架 | LangGraph | 多步驟代理編排 |
| 實驗追蹤 | MLflow | 模型版本、指標記錄 |
| 編排引擎 | Dagster | 資料/ML 管線排程 |
| Chat 前端 | Open WebUI | 使用者互動介面 |
| 監控告警 | Prometheus + Grafana | 基礎設施與服務監控 |
| 持久化儲存 | PostgreSQL + Redis | 狀態儲存與快取 |
| LLM 後端 | 本地 Gemma 4 (Primary) / Azure OpenAI / Anthropic | 本地推論 + 雲端 Fallback（透過 LiteLLM 路由） |

### 3. 理由（Rationale）

**為何本地 GPU + 雲端 Fallback 架構：**
- Primary 使用本地 Gemma 4 26B（4-bit 量化），推論延遲低、資料不離開本地
- PoC 同時驗證「工具鏈整合」與「本地推論可行性」
- 雲端 API 作為 Fallback，確保本地推論異常時仍可運作
- 80%+ 的 Ops 工具不需要 GPU，本地 GPU 專注於 LLM 推論

**為何選擇這些工具：**
- LiteLLM：唯一支援 100+ Provider 的開源 AI Gateway，與 Langfuse 原生整合
- Langfuse：開源、自建部署、OpenTelemetry 原生，社群活躍度最高（22K+ stars）
- Qdrant：Rust 實作、效能優異、Docker 單容器部署、REST + gRPC 雙協議
- NeMo Guardrails：NVIDIA 官方維護，支援 Programmable Guardrails
- LangGraph：LangChain 生態、支援 Human-in-the-Loop、狀態機式 Agent 編排
- MLflow：MLOps 事實標準，Model Registry + Experiment Tracking
- Dagster：Software-Defined Assets 理念，比 Airflow 更現代的 DAG 編排

### 4. 替代方案（Alternatives Considered）

| 替代方案 | 未採用原因 |
|----------|-----------|
| LangSmith 取代 Langfuse | 閉源 SaaS、資料離開本地、費用高 |
| Milvus 取代 Qdrant | 部署複雜度更高（需 etcd + MinIO） |
| Guardrails AI 取代 NeMo | 社群規模較小，企業支援較弱 |
| CrewAI 取代 LangGraph | 抽象層太高，不適合需要精細控制的場景 |
| Airflow 取代 Dagster | 架構較舊，Python-first 體驗不如 Dagster |
| 純雲端 API（無本地推論） | 延遲高、資料離開本地、每次呼叫有費用 |

### 5. 後果（Consequences）

**正面：**
- Docker Compose 一行啟動全棧服務，部署門檻低
- 工具鏈選型可在低成本下充分驗證（僅電費 + 雲端 Fallback 費用）
- 未來升級 GPU 或更換模型只需改 LiteLLM config，不影響其他元件
- 所有資料（含 LLM 推論）保留在本地，滿足資料主權要求

**負面：**
- 需要至少一張 16GB VRAM GPU，有硬體門檻
- 本地推論服務需獨立於 Docker Compose 管理（增加維運複雜度）
- 完整 MLOps 訓練流程需額外 GPU 資源規劃

### 6. 合規與安全考量

- 推論資料透過 NeMo Guardrails 在送出前進行 PII 脫敏
- Langfuse 自建部署，Trace 資料不離開本地
- LiteLLM Virtual Key 隔離各團隊存取權限
- Primary 推論即為本地，已滿足資料主權要求；Fallback 至雲端時由 Guardrails 確保脫敏

#### 6.1 Docker 網路隔離策略

採用三層網路分離設計：

```
┌─────────────────────────────────────────────────────────┐
│ frontend-net（對外）                                      │
│   Open WebUI (:3080), Grafana (:3001), Langfuse (:3100) │
│   MLflow (:5000), LiteLLM (:4000)                       │
├─────────────────────────────────────────────────────────┤
│ backend-net（內部服務通訊）                                │
│   LiteLLM, Langfuse, NeMo Guardrails, Qdrant,          │
│   MLflow, Dagster, Prometheus                           │
├─────────────────────────────────────────────────────────┤
│ data-net（資料層，不對外暴露）                              │
│   PostgreSQL, Redis, ClickHouse, MinIO                  │
└─────────────────────────────────────────────────────────┘
```

| 原則 | 說明 |
|------|------|
| 最小暴露 | 僅 frontend-net 的服務 bind 到 `0.0.0.0`，其餘 bind `127.0.0.1` |
| 資料層隔離 | PostgreSQL / Redis / ClickHouse / MinIO 僅在 data-net 可達 |
| 服務間通訊 | 後端服務透過 backend-net 互通，不經過 frontend-net |
| 密碼管理 | 所有密碼使用 Docker secrets 或 `.env`（不進版控），禁止寫死於 Compose 檔 |
| MinIO Access Key | 透過環境變數注入，Langfuse 與 Mem0 各自使用獨立 bucket 與 Access Key |

---

## 第二部分：技術清單

### 7. 完整元件清單

#### 7.1 核心服務

| # | 元件 | 版本 | Docker Image | Port | 角色 | 依賴 |
|---|------|------|-------------|------|------|------|
| 1 | **LiteLLM Proxy** | latest | `docker.litellm.ai/berriai/litellm:main-latest` | 4000 | AI Gateway | PostgreSQL, Redis |
| 2 | **Langfuse** | v3 | `langfuse/langfuse:latest` | 3100 | LLM 可觀測性 | PostgreSQL, ClickHouse, Redis, MinIO |
| 3 | **Qdrant** | latest | `qdrant/qdrant:latest` | 6333 | 向量資料庫 | 無 |
| 4 | **NeMo Guardrails** | latest | `nvcr.io/nvidia/nemo-guardrails:latest` | 8090 | 安全護欄 | LiteLLM（LLM 呼叫） |
| 5 | **Open WebUI** | latest | `ghcr.io/open-webui/open-webui:main` | 3080 | Chat 前端 | LiteLLM |
| 6 | **MLflow** | latest | `ghcr.io/mlflow/mlflow:latest` | 5000 | 實驗追蹤 / Model Registry | PostgreSQL |
| 7a | **Dagster Webserver** | latest | `dagster/dagster-webserver:latest` | 3070 | 管線編排（UI） | PostgreSQL |
| 7b | **Dagster Daemon** | latest | `dagster/dagster-daemon:latest` | — | 排程執行 / Sensor / Run Queue | PostgreSQL |

#### 7.2 基礎設施

| # | 元件 | Docker Image | Port | 角色 |
|---|------|-------------|------|------|
| 8 | **PostgreSQL** | `postgres:16-alpine` | 5432 | 關聯式資料庫（多服務共用） |
| 9 | **Redis** | `redis:7-alpine` | 6379 | 快取 / 佇列 |
| 10 | **ClickHouse** | `clickhouse/clickhouse-server:latest` | 8123 | Langfuse v3 事件儲存 |
| 11 | **MinIO** | `minio/minio:latest` | 9000 | 物件儲存（Langfuse v3 + Mem0） |
| 12 | **Prometheus** | `prom/prometheus:latest` | 9090 | 指標收集 |
| 13 | **Grafana** | `grafana/grafana:latest` | 3001 | 視覺化 Dashboard |

#### 7.3 開發/測試工具

| # | 元件 | 安裝方式 | 角色 |
|---|------|---------|------|
| 14 | **LangGraph** | pip install | Agent 編排框架 |
| 15 | **LangChain** | pip install | RAG Pipeline 框架 |
| 16 | **Ragas** | pip install | RAG 品質評估 |
| 17 | **DeepEval** | pip install | LLM 輸出評估 |
| 18 | **OpenAI Python SDK** | pip install | 統一 API 呼叫客戶端 |
| 19 | **Mem0** | pip install（自建模式） | Agent 記憶管理（儲存後端：Qdrant + MinIO） |

#### 7.4 LLM 後端

| # | Provider | 模型 | 端點 | 角色 | 月費 |
|---|----------|------|------|------|------|
| 20 | **本地 GPU 推論** | gemma-4-26b-a4b-it-4bit | `http://127.0.0.1:8000/v1` | Primary（低延遲、資料不離開本地） | 電費 |
| 21 | **Azure OpenAI** | GPT-4o | Azure Endpoint | Fallback（穩定） | 按 Token |
| 22 | **Anthropic** | Claude Sonnet | Anthropic API | Fallback（品質） | 按 Token |

### 8. 硬體規格需求

#### 8.1 最低需求

| 資源 | 最低需求 | 建議規格 | 說明 |
|------|---------|---------|------|
| CPU | 8 cores / 16 threads | 16+ cores / 24+ threads | Langfuse ClickHouse 和 Dagster 對 CPU 有一定需求 |
| RAM | **64GB** | **128GB** | 14 個 Docker 服務合計約 30–45GB（ClickHouse ~6GB、PostgreSQL ~3GB、其他各 0.5–2GB），64GB 為最低可用門檻 |
| Disk | 50GB SSD | 100GB+ NVMe SSD | 向量資料庫 + 模型快取 + Docker images 佔用空間 |
| GPU | **必要（VRAM ≥ 16GB）** | VRAM 24GB+（如 RTX 4090） | Gemma 4 26B 4-bit 量化約需 14–16GB VRAM |
| 網路 | 穩定外網連線 | 100Mbps+ | Fallback 時需呼叫雲端 LLM API（Azure / Anthropic） |
| OS | Linux (Ubuntu 22.04+) / macOS / WSL2 | Ubuntu 24.04 LTS | Docker 和 Docker Compose 為必要條件 |

#### 8.2 適用硬體範例

以下為經驗證或預估可行的硬體配置，僅供參考：

| 硬體類型 | 範例規格 | RAM | 適用度 |
|----------|---------|-----|--------|
| 桌上型工作站 | Intel i7/i9 或 AMD Ryzen 7/9 + GPU | 64–128GB | ★★★★★ |
| 迷你工作站 | Intel NUC / GMKtec 等（外接 GPU） | 64GB | ★★★ |
| 筆記型電腦 | 8+ core / 64GB+ RAM + GPU | 64GB | ★★★（散熱受限） |
| 雲端 VM | AWS g5.4xlarge / Azure NC8as_T4_v3 | 64GB | ★★★★★ |
| NAS / Home Server | Intel i5+ / 32GB+（核心子集模式） | 32GB | ★★（僅跑核心子集） |

> **注意**：若 RAM 僅有 32GB，僅可運行**核心子集**：LiteLLM + Langfuse（改用 v2，去掉 ClickHouse + MinIO） + Qdrant + PostgreSQL + Redis。需關閉 Dagster、MLflow、Prometheus、Grafana、NeMo Guardrails、Open WebUI，對應驗證項將無法完成。

#### 8.3 GPU 需求

本方案使用 Gemma 4 26B（4-bit 量化）作為 Primary LLM，GPU 為必要條件：

| GPU VRAM | 適用情境 |
|----------|---------|
| 16GB（如 RTX 4080） | 最低門檻，可運行 gemma-4-26b-a4b-it-4bit，但餘裕有限 |
| 24GB+（如 RTX 4090） | 建議規格，可同時跑 LLM + Embedding 模型 |
| 48GB+（如 A6000） | 可考慮更大模型或多模型同時載入 |

> **注意**：本地推論服務（`http://127.0.0.1:8000/v1`）需在 Docker Compose 啟動前獨立啟動，確保 OpenAI-compatible API 端點可用。

### 9. 網路拓撲

```
                    ┌─── frontend-net（對外 0.0.0.0）────────────────────┐
                    │                                                    │
  User Browser ─────┤  Open WebUI (:3080)    Grafana (:3001)            │
                    │  Langfuse (:3100)      MLflow (:5000)             │
                    │  LiteLLM Proxy (:4000)                            │
                    │                                                    │
                    └────────────────────────┬───────────────────────────┘
                                            │
                    ┌─── backend-net（內部 127.0.0.1）───────────────────┐
                    │                       │                            │
                    │   ┌───────────────────▼───────────────────────┐   │
                    │   │           LiteLLM Proxy                    │   │
                    │   │   ┌─────────┬──────────┬──────────────┐   │   │
                    │   │   │ Route   │ Fallback │ Virtual Key  │   │   │
                    │   │   │ Engine  │ Chain    │ Management   │   │   │
                    │   │   └────┬────┴─────┬────┴──────────────┘   │   │
                    │   └────────┼──────────┼───────────────────────┘   │
                    │            │          │                            │
                    │      ┌─────▼────┐ ┌──▼─────────┐                 │
                    │      │ Local    │ │ Azure /    │  (外部網路)      │
                    │      │ Gemma 4  │ │ Anthropic  │                 │
                    │      │ :8000    │ │ (Fallback) │                 │
                    │      └──────────┘ └────────────┘                 │
                    │                                                    │
                    │   ┌────────────────┐  ┌────────────────────┐      │
                    │   │ Langfuse       │  │ Guardrails (:8090) │      │
                    │   │ (App Server)   │  │ NeMo               │      │
                    │   └────────────────┘  └────────────────────┘      │
                    │                                                    │
                    │   ┌──────┐ ┌──────┐ ┌─────────┐ ┌───────────┐    │
                    │   │Qdrant│ │MLflow│ │ Dagster │ │Prometheus │    │
                    │   │:6333 │ │:5000 │ │ :3070   │ │:9090      │    │
                    │   └──────┘ └──────┘ └─────────┘ └───────────┘    │
                    │                                                    │
                    └────────────────────────┬───────────────────────────┘
                                            │
                    ┌─── data-net（資料層，不對外暴露）──────────────────┐
                    │                       │                            │
                    │   ┌──────────────┐  ┌───────┐  ┌────────────┐    │
                    │   │ PostgreSQL   │  │ Redis │  │ ClickHouse │    │
                    │   │ :5432        │  │ :6379 │  │ :8123      │    │
                    │   └──────────────┘  └───────┘  └────────────┘    │
                    │                                                    │
                    │   ┌──────────────┐                                │
                    │   │ MinIO :9000  │                                │
                    │   └──────────────┘                                │
                    │                                                    │
                    └───────────────────────────────────────────────────┘
```

### 10. 資料流程圖

```
使用者
  │
  ├─[Chat] → Open WebUI (:3080)
  │                │
  │                ▼
  │         LiteLLM Proxy (:4000)
  │           │         │
  │           │    ┌────▼──────────────────────────────┐
  │           │    │ NeMo Guardrails (:8090)           │
  │           │    │                                    │
  │           │    │ 模式 A：應用層攔截                   │
  │           │    │   App/Agent → Guardrails → LiteLLM │
  │           │    │                                    │
  │           │    │ 模式 B：LiteLLM Pre/Post Hook      │
  │           │    │   LiteLLM → Guardrails（自動攔截）  │
  │           │    │                                    │
  │           │    │ → PII 偵測 → 攔截/脫敏              │
  │           │    └────┬──────────────────────────────┘
  │           │         │
  │           │    ┌────▼──────────┐
  │           │    │ 路由決策       │
  │           │    │ Primary:      │── 本地 Gemma 4 (:8000)
  │           │    │ Fallback:     │── Azure OpenAI → Anthropic
  │           │    └────┬──────────┘
  │           │         │
  │           │    ┌────▼──────────┐
  │           │    │ Langfuse      │── Trace 記錄（Token、延遲、成本）
  │           │    │ Callback      │
  │           │    └───────────────┘
  │           │
  │           ▼
  │    Response → Open WebUI → 使用者
  │
  ├─[RAG Query]
  │    App → LangChain → Qdrant (語意檢索)
  │                         │
  │                    Retrieved Docs
  │                         │
  │                    Prompt + Context
  │                         │
  │                    LiteLLM → LLM → Response
  │
  ├─[Agent Task]
  │    App → LangGraph Agent
  │           │
  │           ├─ Step 1: LLM 推論（via LiteLLM）
  │           ├─ Step 2: Tool Call（MCP / API）
  │           ├─ Step 3: LLM 推論（via LiteLLM）
  │           └─ Final: Response
  │           │
  │           └─ 每步 Trace → Langfuse
  │
  └─[ML Experiment]
       Dagster Pipeline → 資料處理 → 模型訓練/評估
                                         │
                                    MLflow 記錄指標 + 模型版本
```

---

## 第三部分：工作清單

### 11. PoC 驗證目標（25 項）

| 編號 | 領域 | 驗證項目 | 成功標準 |
|------|------|---------|---------|
| V-01 | LLMOps | LiteLLM → 本地 Gemma 4 推論 | 端到端回應正常，Streaming 正常 |
| V-02 | LLMOps | LiteLLM → Azure OpenAI 推論 | 回應正常，模型切換無感 |
| V-03 | LLMOps | LiteLLM Fallback 機制 | 本地推論異常時自動切至 Azure |
| V-04 | LLMOps | LiteLLM Virtual Key 認證 | 無效 Key 回傳 401，超額回傳 429 |
| V-05 | LLMOps | LiteLLM 成本追蹤 | Dashboard 顯示各 Key 用量與費用 |
| V-06 | LLMOps | Langfuse Trace 收集 | 每個 LLM 呼叫產生完整 Trace |
| V-07 | LLMOps | Langfuse 延遲分析 | 可查看 TTFT、總延遲、Token 數 |
| V-08 | LLMOps | Langfuse Prompt 管理 | 可建立/版控/部署 Prompt Template |
| V-09 | LLMOps | Langfuse Eval 整合 | 用 Ragas 對 RAG 回答做品質評分 |
| V-10 | LLMOps | Qdrant 向量儲存 | 文件 Embedding 寫入與語意查詢 |
| V-11 | LLMOps | RAG 端到端流程 | 文件上傳 → 向量化 → 檢索 → 生成回答 |
| V-12 | LLMOps | NeMo Guardrails PII 過濾 | 包含身分證/信用卡號的輸入被攔截 |
| V-13 | LLMOps | NeMo Guardrails 主題限制 | 限定模型只回答特定領域問題 |
| V-14 | LLMOps | Open WebUI Chat | 使用者可透過 UI 與模型對話 |
| V-15 | AgentOps | LangGraph 單 Agent 執行 | Agent 完成多步驟任務（搜尋 → 分析 → 回答） |
| V-16 | AgentOps | LangGraph Tool Calling | Agent 呼叫外部工具（計算機/API）成功 |
| V-17 | AgentOps | LangGraph Human-in-the-Loop | 關鍵步驟暫停等待人工確認 |
| V-18 | AgentOps | Agent Trace in Langfuse | Agent 每步決策可在 Langfuse 追蹤 |
| V-19 | AgentOps | Agent 記憶管理 (Mem0 自建) | Agent 跨對話保留上下文記憶，向量存 Qdrant、檔案存 MinIO |
| V-20 | MLOps | MLflow 實驗記錄 | 記錄超參數、指標、模型版本 |
| V-21 | MLOps | MLflow Model Registry | 模型註冊、Stage 切換（Staging → Production） |
| V-22 | MLOps | Dagster Pipeline 執行 | DAG 排程執行資料處理任務 |
| V-23 | Infra | Prometheus 指標收集 | 收集 LiteLLM + Langfuse + 系統指標 |
| V-24 | Infra | Grafana Dashboard | 視覺化延遲、Token 用量、錯誤率、成本 |
| V-25 | Infra | 全棧 Docker Compose 啟停 | 一行指令啟動/停止所有 14 個服務 |

### 12. 工作清單與時程（4 週 / 20 個工作天）

> **執行方式**：本工作清單以 Coding Agent 自動執行為主，時程為參考估計。各 Phase 依序完成，Phase 內的任務可由 Agent 平行處理。
>
> **時程說明**：天數為參考值，實際進度取決於 Agent 執行效率與除錯時間。

#### Phase 1：基礎建置 & AI Gateway（Day 1–3）

| # | 任務 | 天 | 產出物 | 驗證項 |
|---|------|----|--------|--------|
| W1.1 | Docker Compose 骨架建立（PostgreSQL, Redis） | D1 | docker-compose.yml | — |
| W1.2 | LiteLLM Proxy 部署 + config.yaml 設定 | D1 | LiteLLM :4000 啟動 | — |
| W1.3 | 本地 Gemma 4 推論服務啟動確認（`http://127.0.0.1:8000/v1`） | D1 | 本地推論端點可用 | — |
| W1.4 | LiteLLM → 本地 Gemma 4 推論驗證 | D2 | curl 測試結果 | **V-01** |
| W1.5 | LiteLLM → Azure OpenAI 設定 + 驗證 | D2 | 多模型路由成功 | **V-02** |
| W1.6 | LiteLLM Fallback Chain 設定 + 驗證（模擬本地推論停機） | D3 | 切換至 Azure 成功 | **V-03** |
| W1.7 | Virtual Key / Team / Budget / Rate Limit 設定 + 驗證 | D3 | 無效 Key 回傳 401，超額回傳 429 | **V-04** |
| W1.8 | 成本追蹤 Dashboard 確認 | D3 | LiteLLM UI 截圖 | **V-05** |

#### Phase 2：可觀測性 & RAG Pipeline（Day 4–8）

| # | 任務 | 天 | 產出物 | 驗證項 |
|---|------|----|--------|--------|
| W2.1 | Langfuse v3 Docker Compose 部署（含 ClickHouse + MinIO）；若失敗則降級至 v2（僅需 PostgreSQL） | D4 | Langfuse :3100 啟動 | — |
| W2.2 | LiteLLM → Langfuse Callback 設定 | D4 | config.yaml 加入 callback | — |
| W2.3 | Trace 收集驗證 + 延遲分析（TTFT、Token 數） | D5 | Langfuse Trace 截圖 | **V-06**, **V-07** |
| W2.4 | Prompt 管理功能測試 | D5 | 建立 Prompt Template | **V-08** |
| W2.5 | Qdrant Docker 部署 | D6 | Qdrant :6333 啟動 | — |
| W2.6 | Python 腳本：文件 Embedding + 寫入 Qdrant | D6 | Embedding 成功 | **V-10** |
| W2.7 | Python 腳本：語意查詢驗證 | D7 | 查詢返回相關文件 | — |
| W2.8 | LangChain RAG 端到端 Pipeline | D7 | 完整 RAG 回答 | **V-11** |
| W2.9 | Ragas RAG 品質評估 | D8 | 評分報告 | **V-09** |

#### Phase 3：安全護欄 & Chat 前端（Day 9–11）

| # | 任務 | 天 | 產出物 | 驗證項 |
|---|------|----|--------|--------|
| W3.1 | NeMo Guardrails 部署 + 設定檔 | D9 | Guardrails :8090 啟動 | — |
| W3.2 | 模式 A 驗證：應用層呼叫 Guardrails → PII 過濾（身分證號、信用卡號） | D10 | 攔截成功 | **V-12** |
| W3.3 | 模式 B 驗證：LiteLLM Pre/Post Hook → 主題限制（Off-topic 拒絕回答） | D10 | 限制成功 | **V-13** |
| W3.4 | Open WebUI 部署 + 連接 LiteLLM | D11 | Chat UI 可用 | **V-14** |
| W3.5 | Phase 1–3 驗證結果彙整 | D11 | 14/25 項驗證紀錄 | — |

#### Phase 4：AgentOps（Day 12–15）

| # | 任務 | 天 | 產出物 | 驗證項 |
|---|------|----|--------|--------|
| W4.1 | Python 虛擬環境 + LangGraph 安裝 | D12 | venv 就緒 | — |
| W4.2 | 單 Agent 範例（ReAct 模式） | D12 | Agent 多步驟推論成功 | **V-15** |
| W4.3 | Agent Tool Calling（計算機/天氣 API） | D12 | 工具呼叫成功 | **V-16** |
| W4.4 | LangGraph Human-in-the-Loop 實作 | D13 | 暫停等待人工確認 | **V-17** |
| W4.5 | Agent Trace → Langfuse 串接 | D13 | 每步決策可追蹤 | **V-18** |
| W4.6 | Mem0 自建部署（向量存 Qdrant、檔案存 MinIO）+ 記憶管理整合 | D14 | 跨對話記憶保留 | **V-19** |
| W4.7 | Agent + RAG Tool（從 Qdrant 檢索） | D14 | Agent 使用 RAG 回答問題 | — |
| W4.8 | Agent + Guardrails（輸出過濾） | D15 | 不當回答被攔截 | — |
| W4.9 | 全鏈路 Trace 驗證（Agent → RAG → LLM → Guardrails） | D15 | Langfuse 完整鏈路 | — |

#### Phase 5：MLOps（Day 16–17）

| # | 任務 | 天 | 產出物 | 驗證項 |
|---|------|----|--------|--------|
| W5.1 | MLflow Docker 部署 | D16 | MLflow :5000 啟動 | — |
| W5.2 | 實驗記錄測試（sklearn 模型） | D16 | 超參數+指標記錄成功 | **V-20** |
| W5.3 | Model Registry 測試 | D16 | 模型註冊 + Stage 切換 | **V-21** |
| W5.4 | Dagster Docker 部署（含 dagster-daemon） | D17 | Dagster :3070 啟動 | — |
| W5.5 | 範例 Pipeline（資料處理 → 特徵工程 → 評估） | D17 | DAG 執行成功 | **V-22** |
| W5.6 | Dagster + MLflow 整合 | D17 | Pipeline 結果寫入 MLflow | — |

#### Phase 6：監控 & Docker Compose 整合（Day 18–19）

| # | 任務 | 天 | 產出物 | 驗證項 |
|---|------|----|--------|--------|
| W6.1 | Prometheus 部署 + scrape config | D18 | Prometheus :9090 啟動 | — |
| W6.2 | Grafana 部署 + 資料來源設定 | D18 | Grafana :3001 啟動 | — |
| W6.3 | LiteLLM /metrics → Prometheus | D18 | 指標收集成功 | **V-23** |
| W6.4 | Grafana Dashboard 建置 | D18 | 4 個面板就緒 | **V-24** |
| W6.5 | 統一 docker-compose.yml 整合所有服務 | D19 | 一行啟動 14 個服務 | **V-25** |
| W6.6 | 健康檢查（healthcheck）設定 | D19 | 所有服務 healthy | — |
| W6.7 | .env 檔案整理（所有密鑰/設定） | D19 | 環境變數集中管理 | — |
| W6.8 | 啟動/停止/重啟 SOP | D19 | README.md | — |

#### Phase 7：整合測試 & 收尾（Day 20 + Buffer）

| # | 任務 | 天 | 產出物 | 驗證項 |
|---|------|----|--------|--------|
| W7.1 | 全鏈路整合測試腳本 | D20 | test_integration.py | — |
| W7.2 | 25 項驗證目標逐一確認 | D20 | 驗證清單 25/25 | — |
| W7.3 | 輕量壓力測試（10 併發 Chat） | D20 | 延遲 + 錯誤率記錄 | — |
| W7.4 | PoC 成果報告 | Buffer | poc-report.md | — |
| W7.5 | 已知問題與限制清單 | Buffer | known-issues.md | — |
| W7.6 | 生產部署建議（GPU 規格 + 架構調整） | Buffer | production-plan.md | — |
| W7.7 | PoC Demo 準備 + 展示 | Buffer | 簡報 + Live Demo | — |
| W7.8 | 工具鏈最終選型決策 | Buffer | ADR-001 狀態更新為 Accepted/Rejected | — |

### 13. Grafana Dashboard 規劃

| Dashboard | 面板內容 |
|-----------|---------|
| **LLMOps Overview** | 總請求數、平均延遲、錯誤率、Token 總消耗、各 Provider 請求分佈、Fallback 觸發次數 |
| **Cost Analytics** | 各 Team 費用、各模型費用、每日趨勢、Budget 使用率 |
| **RAG Performance** | Qdrant 查詢延遲、Embedding 吞吐量、RAG Eval 分數趨勢 |
| **System Health** | CPU / RAM 使用率、Docker 容器狀態、各服務 Uptime |

### 14. PoC 評估矩陣

PoC 結束後依以下維度評分（1–5 分），各維度加權後滿分為 **5.0 分**，**加權總分 ≥ 4.0 即建議進入生產部署階段**：

> **計算方式**：加權總分 = Σ（各維度分數 × 權重），例如全部 5 分 → 5.0，全部 3 分 → 3.0

| # | 評估維度 | 權重 | 1 分（不可接受） | 3 分（基本達標） | 5 分（優秀） |
|---|---------|------|-----------------|-----------------|-------------|
| 1 | 驗證完成度 | 15% | 達成率 < 60%（< 15/25） | 達成率 80%（20/25） | 達成率 100%（25/25） |
| 2 | LiteLLM 路由穩定性 | 15% | Fallback 切換 > 30s 或錯誤率 > 5% | 切換 < 10s，錯誤率 < 2% | 切換 < 3s，錯誤率 < 0.5% |
| 3 | Langfuse 追蹤完整度 | 10% | Trace 覆蓋率 < 50% | 覆蓋率 80%+，欄位基本完整 | 覆蓋率 100%，含 Token/延遲/成本全欄位 |
| 4 | RAG 品質 | 10% | Ragas Faithfulness < 0.5 | Faithfulness ≥ 0.7，Relevancy ≥ 0.7 | Faithfulness ≥ 0.9，Relevancy ≥ 0.9 |
| 5 | Guardrails 有效性 | 10% | PII 攔截率 < 70% 或 FP > 20% | 攔截率 ≥ 90%，FP < 10% | 攔截率 ≥ 99%，FP < 3% |
| 6 | Agent 可靠性 | 10% | 任務完成率 < 60% | 完成率 ≥ 80%，平均 < 10 步 | 完成率 ≥ 95%，平均 < 5 步 |
| 7 | MLOps 工具完整度 | 5% | 僅能記錄指標，無 Registry | 指標 + Registry + Stage 切換皆可用 | 完整 Pipeline（Dagster → MLflow）自動化 |
| 8 | 監控可觀測性 | 10% | Prometheus 有資料但無 Dashboard | 4 個 Dashboard 可用，手動告警 | Dashboard + 自動告警規則（Slack/Email） |
| 9 | 部署易用度 | 10% | 啟動需 > 10 分鐘或需手動介入 | 一行啟動，< 5 分鐘，有 README | 一行啟動 < 3 分鐘，含 healthcheck + 自動重試 |
| 10 | 團隊可維護性 | 5% | 無文件，依賴個人知識 | 有基本操作文件，社群活躍 | 完整 Runbook + Troubleshooting Guide |

---

## 第四部分：附錄

### 附錄 A：關鍵環境變數清單

```bash
# .env 檔案

# ── LiteLLM ──
LITELLM_MASTER_KEY=sk-litellm-master-xxxxx
LITELLM_DATABASE_URL=postgresql://litellm:password@postgres:5432/litellm

# ── 本地 Gemma 4 推論 ──
LOCAL_LLM_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_LLM_MODEL=gemma-4-26b-a4b-it-4bit

# ── Azure OpenAI ──
AZURE_API_KEY=xxxxx
AZURE_API_BASE=https://your-endpoint.openai.azure.com/
AZURE_API_VERSION=2024-10-21

# ── Anthropic ──
ANTHROPIC_API_KEY=sk-ant-xxxxx

# ── Langfuse ──
LANGFUSE_SECRET_KEY=sk-lf-xxxxx
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxx
LANGFUSE_HOST=http://langfuse:3100
LANGFUSE_DATABASE_URL=postgresql://langfuse:password@postgres:5432/langfuse

# ── Qdrant ──
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# ── MLflow ──
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:password@postgres:5432/mlflow

# ── MinIO（Langfuse + Mem0）──
MINIO_ROOT_USER=minio-admin
MINIO_ROOT_PASSWORD=<generate-strong-password>
LANGFUSE_S3_BUCKET=langfuse
LANGFUSE_S3_ACCESS_KEY=<generate-access-key>
LANGFUSE_S3_SECRET_KEY=<generate-secret-key>
MEM0_S3_BUCKET=mem0
MEM0_S3_ACCESS_KEY=<generate-access-key>
MEM0_S3_SECRET_KEY=<generate-secret-key>

# ── Grafana ──
GF_SECURITY_ADMIN_PASSWORD=<generate-strong-password>
```

### 附錄 B：快速啟動指令

```bash
# 1. Clone PoC 專案
git clone https://github.com/your-org/llmops-poc.git
cd llmops-poc

# 2. 設定環境變數
cp .env.example .env
# 編輯 .env 填入 API Key

# 3. 啟動本地推論服務（需先確認 GPU 可用）
# 例如使用 vLLM:
# vllm serve gemma-4-26b-a4b-it-4bit --port 8000
# 確認端點可用:
curl http://127.0.0.1:8000/v1/models

# 4. 一行啟動全棧（Ops 工具鏈）
docker compose up -d

# 5. 確認所有服務健康
docker compose ps
# 所有服務應顯示 healthy

# 6. 存取各服務
# Open WebUI:  http://localhost:3080
# LiteLLM:     http://localhost:4000/ui
# Langfuse:    http://localhost:3100
# MLflow:      http://localhost:5000
# Dagster:     http://localhost:3070
# Grafana:     http://localhost:3001
# Qdrant:      http://localhost:6333/dashboard

# 7. 快速測試
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-litellm-master-xxxxx" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-26b","messages":[{"role":"user","content":"Hello"}]}'

# 8. 停止全棧
docker compose down
```

### 附錄 C：參考資源

| 元件 | 文件連結 |
|------|---------|
| LiteLLM | https://docs.litellm.ai/ |
| Langfuse | https://langfuse.com/docs |
| Langfuse Docker Compose | https://langfuse.com/self-hosting/docker-compose |
| LiteLLM + Langfuse 整合 | https://langfuse.com/docs/integrations/litellm |
| Qdrant | https://qdrant.tech/documentation/ |
| NeMo Guardrails | https://docs.nvidia.com/nemo/guardrails/ |
| LangGraph | https://langchain-ai.github.io/langgraph/ |
| MLflow | https://mlflow.org/docs/latest/ |
| Dagster | https://docs.dagster.io/ |
| Open WebUI | https://docs.openwebui.com/ |
| Ragas (RAG Eval) | https://docs.ragas.io/ |
| DeepEval | https://docs.confident-ai.com/ |
| Mem0 | https://docs.mem0.ai/ |
| Gemma 4 (Google) | https://ai.google.dev/gemma |
| vLLM | https://docs.vllm.ai/ |

### 附錄 D：風險登記冊

| # | 風險 | 可能性 | 影響 | 緩解措施 |
|---|------|--------|------|---------|
| R1 | 本地 GPU 推論服務異常（OOM、硬體故障） | 中 | 中 | LiteLLM Fallback 至 Azure/Anthropic；監控 GPU 記憶體使用率 |
| R2 | Langfuse v3 部署複雜（需 ClickHouse + MinIO） | 中 | 中 | 降級方案：退回 Langfuse v2（僅需 PostgreSQL），功能不受影響僅效能較低 |
| R3 | NeMo Guardrails 對非 OpenAI 模型相容性問題 | 中 | 中 | 透過 LiteLLM 統一為 OpenAI 格式 |
| R4 | Docker Compose 記憶體不足（64GB 以下） | 中 | 高 | 最低 64GB；32GB 環境僅跑核心子集（LiteLLM + Langfuse v2 + Qdrant） |
| R5 | 雲端 API Key 洩漏 | 低 | 高 | .env 不進版控、設定 Budget 上限 |
| R6 | Qdrant 向量維度不匹配 | 低 | 低 | 統一使用同一 Embedding 模型 |
| R7 | LiteLLM 供應鏈安全問題 | 低 | 高 | 使用最新版本、監控安全公告 |
