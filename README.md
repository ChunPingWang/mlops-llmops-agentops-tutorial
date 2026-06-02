# LLMOps / AgentOps / MLOps PoC 全棧工具鏈教學

> 一站式部署 14 個服務，涵蓋 LLM 路由、可觀測性、RAG、安全護欄、Agent 編排、ML 實驗追蹤與監控告警。

---

## 目錄

- [專案概覽](#專案概覽)
- [架構總覽](#架構總覽)
- [技術原理詳解](#技術原理詳解)
- [目錄結構](#目錄結構)
- [快速開始](#快速開始)
- [設定檔詳解](#設定檔詳解)
- [腳本詳解](#腳本詳解)
- [驗證項目清單](#驗證項目清單)
- [常見問題](#常見問題)

---

## 專案概覽

### 這個專案在做什麼？

這是一個 **PoC（概念驗證）專案**，目的是在本地環境搭建一套完整的 AI 維運平台，驗證以下三大領域的工具鏈整合：

| 領域 | 全名 | 簡單解釋 |
|------|------|---------|
| **LLMOps** | Large Language Model Operations | 管理和監控大型語言模型的運維流程 |
| **AgentOps** | Agent Operations | 管理自主 AI 代理的生命週期和追蹤 |
| **MLOps** | Machine Learning Operations | 機器學習模型的訓練、部署和版本管理 |

### 為什麼需要這些？

當你把 AI 從「玩具」變成「產品」時，需要解決：
- **模型路由**：同時使用多個 LLM（本地 + 雲端），故障時自動切換
- **成本控制**：追蹤每個 API 呼叫花了多少錢
- **品質監控**：每個回答的延遲、Token 數、品質分數
- **安全防護**：防止用戶輸入敏感資料（信用卡號）、防止模型回答不當內容
- **記憶管理**：讓 AI Agent 記住跨對話的上下文
- **實驗管理**：追蹤模型訓練的每次實驗、比較結果

---

## 架構總覽

### 系統架構圖

```
使用者 (瀏覽器)
    │
    ├── Open WebUI (:3080)     ← Chat 介面
    ├── Grafana (:3001)        ← 監控 Dashboard
    ├── Langfuse (:3100)       ← LLM 追蹤介面
    └── MLflow (:5000)         ← 實驗追蹤介面
         │
         ▼
┌─── frontend-net ──────────────────────────────────────────┐
│                                                            │
│   LiteLLM Proxy (:4000)  ← 所有 LLM 請求的入口           │
│       │                                                    │
│       ├─→ 本地 Gemma 4 (:8000)   [Primary - 低延遲]       │
│       ├─→ Azure OpenAI            [Fallback - 穩定]        │
│       └─→ Anthropic Claude        [Fallback - 品質]        │
│                                                            │
├─── backend-net ───────────────────────────────────────────┤
│                                                            │
│   Langfuse        ← 追蹤每一次 LLM 呼叫                    │
│   NeMo Guardrails ← 過濾敏感資料、限制主題                  │
│   Qdrant          ← 儲存文件向量（用於 RAG）                │
│   Prometheus      ← 收集所有服務的指標                      │
│   Dagster         ← 排程執行 ML Pipeline                   │
│                                                            │
├─── data-net ──────────────────────────────────────────────┤
│                                                            │
│   PostgreSQL  ← 關聯式資料庫（存設定、使用者、實驗紀錄）      │
│   Redis       ← 快取和佇列                                 │
│   ClickHouse  ← 高速分析型資料庫（Langfuse 事件）           │
│   MinIO       ← 物件儲存（檔案、模型產物）                   │
│                                                            │
└───────────────────────────────────────────────────────────┘
```

### 三層網路隔離

| 網路層 | 用途 | 暴露方式 |
|--------|------|---------|
| `frontend-net` | 使用者可以直接存取的服務 | 綁定 `0.0.0.0`（對外開放） |
| `backend-net` | 服務之間的內部通訊 | 僅容器間可達 |
| `data-net` | 資料庫和儲存層 | 綁定 `127.0.0.1`（僅本機） |

**為什麼要分層？** 假設你的電腦在公司網路中，分層可以防止外部直接存取資料庫。即使 Open WebUI 被入侵，攻擊者也無法直接連到 PostgreSQL。

---

## 技術原理詳解

### 1. LiteLLM — AI Gateway（AI 閘道器）

**是什麼？** 一個統一的 API 代理，讓你用同一個介面呼叫不同的 LLM。

**為什麼需要？** 想像你有 3 家外送平台的帳號（Gemma、GPT、Claude），LiteLLM 就像一個統一的 App，你只需要跟它說「幫我叫外送」，它會根據規則決定用哪家平台。

**核心功能：**

```
你的應用程式
    │
    │  統一的 OpenAI 格式 API
    ▼
┌─────────────────────────────────┐
│         LiteLLM Proxy           │
│                                 │
│  ┌─────────┐  ┌──────────────┐ │
│  │ 路由引擎 │  │ Fallback 鏈  │ │
│  │         │  │              │ │
│  │ gemma → │  │ gemma 失敗？ │ │
│  │ azure → │  │ → 試 azure   │ │
│  │ claude→ │  │ → 試 claude  │ │
│  └─────────┘  └──────────────┘ │
│                                 │
│  ┌─────────────────────────┐   │
│  │ Virtual Key Management  │   │
│  │ 每個團隊一把 Key        │   │
│  │ 設定用量上限和預算       │   │
│  └─────────────────────────┘   │
└─────────────────────────────────┘
```

**Fallback 機制：** 當本地 Gemma 4 回應逾時或出錯時，LiteLLM 會自動把請求轉發給 Azure OpenAI。你的應用程式完全不需要知道這件事。

---

### 2. Langfuse — LLM 可觀測性

**是什麼？** LLM 世界的「監視器」，記錄每一次 LLM 呼叫的完整細節。

**為什麼需要？** 如果你的 AI 回答品質下降了，你需要知道：
- 是模型變慢了？（延遲分析）
- 是 Token 用太多了？（成本分析）
- 是某個 Prompt 寫壞了？（Prompt 版本管理）

**追蹤的資料：**

```
一個 Trace（追蹤）的結構：
├── Session: user-123
├── Trace: "使用者問了一個問題"
│   ├── Span: "RAG 檢索" (耗時 200ms)
│   │   └── 從 Qdrant 取得 3 篇文件
│   ├── Generation: "LLM 呼叫" (耗時 1500ms)
│   │   ├── Model: gemma-4-26b
│   │   ├── Input tokens: 856
│   │   ├── Output tokens: 234
│   │   ├── Cost: $0.0012
│   │   └── TTFT: 180ms（第一個 Token 的延遲）
│   └── Span: "Guardrails 檢查" (耗時 50ms)
│       └── Result: PASS
└── Score: faithfulness = 0.85
```

**TTFT (Time to First Token)：** 使用者按下送出後，看到第一個字出現的時間。這是最影響使用體驗的指標。

---

### 3. Qdrant — 向量資料庫

**是什麼？** 一個專門儲存和搜尋「向量」的資料庫。

**什麼是向量？** 把文字轉換成一串數字（通常 768 或 1536 個），讓電腦能理解語意相似度。

```
"人工智慧很強大" → [0.23, -0.45, 0.67, ..., 0.12]  (768 個數字)
"AI 非常厲害"    → [0.21, -0.43, 0.65, ..., 0.14]  (很接近！)
"今天天氣很好"   → [0.89, 0.12, -0.34, ..., 0.56]  (很不一樣)
```

**用途（RAG - Retrieval-Augmented Generation）：**

```
1. 文件切片：把 PDF 切成小段落
2. 向量化：每個段落 → Embedding 模型 → 向量
3. 儲存：向量 + 原文存入 Qdrant
4. 檢索：使用者問題 → 向量 → 找到最相似的段落
5. 生成：把找到的段落 + 問題一起送給 LLM
```

**為什麼不直接把整份 PDF 給 LLM？** 因為 LLM 有 Token 上限（也更貴），而且找到最相關的段落比整份文件更能產生精確的回答。

---

### 4. NeMo Guardrails — 安全護欄

**是什麼？** NVIDIA 開發的安全框架，在 LLM 的輸入和輸出端加上「過濾器」。

**兩種模式：**

```
模式 A：應用層攔截（攔截在前）
┌──────────────┐    ┌───────────────┐    ┌─────────┐
│ 使用者輸入    │ →  │ Guardrails    │ →  │ LiteLLM │
│ "我的信用卡是 │    │ 偵測到 PII！   │    │ (不會到) │
│  4111-..."   │    │ → 攔截 + 拒絕  │    └─────────┘
└──────────────┘    └───────────────┘

模式 B：LiteLLM Hook（攔截在後）
┌──────────────┐    ┌─────────┐    ┌───────────────┐
│ 使用者問：    │ →  │ LiteLLM │ →  │ Guardrails    │
│ "怎麼做炸彈" │    │ 產生回答  │    │ 偵測到違規！   │
└──────────────┘    └─────────┘    │ → 攔截回答     │
                                    └───────────────┘
```

**PII（個人識別資訊）：** 信用卡號、身分證號、社會安全碼等。Guardrails 會偵測這些模式並阻止處理。

---

### 5. LangGraph — Agent 框架

**是什麼？** LangChain 生態的 Agent 編排框架，用「有向圖」來定義 AI Agent 的行為流程。

**什麼是 Agent？** 一個能自主決定下一步行動的 AI。不只是「問問題→得到回答」，而是「分析問題→決定用什麼工具→執行→看結果→決定下一步」。

**ReAct 模式（Reasoning + Acting）：**

```
使用者："144 的平方根加上今年是幾年？"

Agent 思考流程：
├── Step 1: 我需要計算 √144
│   └── 呼叫 calculator 工具 → 12
├── Step 2: 我需要知道今年
│   └── 呼叫 get_current_time 工具 → 2026
├── Step 3: 12 + 2026 = 2038
└── Final: "144 的平方根是 12，加上今年 2026 等於 2038"
```

**Human-in-the-Loop（人機協作）：**

```
Agent 執行流程：
├── research: LLM 產生行動計劃
├── human_review: ⏸️ 暫停！等待人工確認
│   └── 人類審查計劃 → "approve" / "reject"
└── execute: 獲得批准後繼續執行
```

這在高風險場景很重要：例如 Agent 要發送郵件或修改資料庫時，先讓人確認。

---

### 6. Mem0 — Agent 記憶管理

**是什麼？** 讓 AI Agent 擁有跨對話的長期記憶。

**為什麼需要？** 普通的 ChatGPT 對話結束後就忘記了。但如果你的 AI 助理需要記住「用戶偏好的程式語言」、「上次討論到的專案進度」，就需要 Mem0。

```
對話 1（今天）：
  使用者："我叫 Alice，在 TechCorp 工作"
  → Mem0 儲存到 Qdrant: {user: alice, fact: "works at TechCorp"}

對話 2（下週）：
  使用者："我在哪裡工作？"
  → Mem0 從 Qdrant 搜尋 user=alice 的記憶
  → 找到: "works at TechCorp"
  → Agent: "你在 TechCorp 工作"
```

---

### 7. MLflow — 實驗追蹤與模型管理

**是什麼？** ML 界的「實驗筆記本」+ 「模型倉庫」。

**為什麼需要？** 訓練模型時你會嘗試不同的超參數：

```
實驗 1: n_estimators=100, max_depth=5  → accuracy=0.92
實驗 2: n_estimators=200, max_depth=10 → accuracy=0.95  ← 最好！
實驗 3: n_estimators=300, max_depth=15 → accuracy=0.93  (過擬合)
```

MLflow 自動記錄這些，讓你比較和重現結果。

**Model Registry（模型註冊表）：**

```
iris-classifier
├── Version 1 → Stage: Archived（舊版）
├── Version 2 → Stage: Staging（測試中）
└── Version 3 → Stage: Production（正式上線）
```

---

### 8. Dagster — 管線編排

**是什麼？** 一個現代化的工作流程排程引擎。

**為什麼需要？** ML 工作流程有明確的步驟順序：

```
┌──────────┐    ┌────────────────┐    ┌──────────────┐
│ raw_data │ →  │ feature_engin. │ →  │ model_eval   │
│ 載入資料  │    │ 特徵工程        │    │ 訓練＋評估    │
└──────────┘    └────────────────┘    └──────────────┘
                                            │
                                            ▼
                                      ┌──────────┐
                                      │ MLflow   │
                                      │ 記錄結果  │
                                      └──────────┘
```

Dagster 確保：步驟按順序執行、失敗時重試、可以排程（每天凌晨跑一次）。

---

### 9. Prometheus + Grafana — 監控告警

**Prometheus** 是什麼？一個時間序列資料庫，每 15 秒去「抓」各服務的指標。

**Grafana** 是什麼？把 Prometheus 的數據畫成圖表的工具。

```
LiteLLM (:4000/metrics)
    │ 每 15 秒
    ▼
Prometheus (儲存指標)
    │
    ▼
Grafana (視覺化)
    ├── 總請求數
    ├── 平均延遲
    ├── 錯誤率
    ├── Token 消耗
    └── Fallback 觸發次數
```

---

### 10. 支援元件

| 元件 | 角色 | 簡單比喻 |
|------|------|---------|
| **PostgreSQL** | 關聯式資料庫 | Excel 試算表（有欄位、有關聯） |
| **Redis** | 快取 / 佇列 | 便利貼（快速暫存常用資料） |
| **ClickHouse** | 分析型資料庫 | 大量日誌的高速搜尋引擎 |
| **MinIO** | 物件儲存 | 自建的 Google Drive（存檔案） |

---

## 目錄結構

```
ml-llm-agent-ops-tutorial/
│
├── ADR-001-LLMOps-AgentOps-MLOps-PoC-Plan.md  # 架構決策文件
├── docker-compose.yml                          # 14 個服務的部署定義
├── .env.example                                # 環境變數範本
├── .gitignore                                  # Git 排除規則
├── requirements.txt                            # Python 依賴套件
│
├── configs/                                    # 所有服務的設定檔
│   ├── litellm/
│   │   └── config.yaml                         # LLM 路由 + Fallback 設定
│   ├── postgres/
│   │   └── init-databases.sh                   # 資料庫初始化腳本
│   ├── prometheus/
│   │   └── prometheus.yml                      # 指標收集目標
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   └── llmops-overview.json            # LLMOps 監控面板
│   │   └── provisioning/
│   │       ├── datasources/datasource.yml      # 資料來源設定
│   │       └── dashboards/dashboard.yml        # Dashboard 載入設定
│   └── guardrails/
│       ├── config.yml                          # Guardrails 主設定
│       ├── prompts.yml                         # 安全檢查 Prompt
│       └── rails/
│           ├── pii.co                          # PII 過濾規則
│           └── topic.co                        # 主題限制規則
│
├── agents/                                     # AI Agent 範例
│   ├── react_agent.py                          # ReAct 多步驟推論 Agent
│   ├── hitl_agent.py                           # Human-in-the-Loop Agent
│   ├── rag_agent.py                            # RAG + Guardrails 全鏈路 Agent
│   └── mem0_agent.py                           # 跨對話記憶管理
│
├── rag/                                        # RAG Pipeline
│   ├── rag_pipeline.py                         # 端到端 RAG 流程
│   └── eval_ragas.py                           # RAG 品質評估
│
├── mlops/                                      # MLOps 工具
│   ├── mlflow_experiment.py                    # MLflow 實驗追蹤
│   ├── dagster_pipeline.py                     # Dagster 管線編排
│   └── workspace.yaml                          # Dagster 工作區設定
│
└── scripts/                                    # 工具腳本
    ├── seed_qdrant.py                          # Qdrant 向量資料庫初始化
    └── test_integration.py                     # 25 項驗證自動化測試
```

---

## 快速開始

### 前置需求

| 項目 | 最低要求 |
|------|---------|
| Docker + Docker Compose | v24+ |
| Python | 3.11+ |
| GPU | VRAM ≥ 16GB（跑 Gemma 4 26B 4-bit） |
| RAM | 64GB |
| Disk | 50GB+ SSD |

### 步驟

```bash
# 1. Clone 專案
git clone https://github.com/ChunPingWang/mlops-llmops-agentops-tutorial.git
cd mlops-llmops-agentops-tutorial

# 2. 設定環境變數
cp .env.example .env
# 編輯 .env，填入你的 API Key（Azure、Anthropic）

# 3. 啟動本地推論服務（需要 GPU）
# 使用 vLLM:
vllm serve gemma-4-26b-a4b-it-4bit --port 8000
# 確認可用:
curl http://127.0.0.1:8000/v1/models

# 4. 啟動全棧服務
docker compose up -d

# 5. 確認健康狀態
docker compose ps

# 6. 安裝 Python 依賴（跑腳本用）
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 7. 初始化向量資料庫
python scripts/seed_qdrant.py

# 8. 執行整合測試
python scripts/test_integration.py
```

### 服務存取

| 服務 | URL | 用途 |
|------|-----|------|
| Open WebUI | http://localhost:3080 | Chat 介面 |
| LiteLLM | http://localhost:4000/ui | API Gateway 管理 |
| Langfuse | http://localhost:3100 | LLM 追蹤 |
| MLflow | http://localhost:5000 | 實驗追蹤 |
| Dagster | http://localhost:3070 | 管線排程 |
| Grafana | http://localhost:3001 | 監控 Dashboard |
| Qdrant | http://localhost:6333/dashboard | 向量 DB 管理 |

---

## 設定檔詳解

### docker-compose.yml

這是整個專案的核心，定義了 14 個 Docker 容器如何啟動和互連。

**關鍵概念：**

```yaml
services:
  litellm:
    image: docker.litellm.ai/berriai/litellm:main-latest  # 使用的映像
    ports:
      - "0.0.0.0:4000:4000"    # 對外開放（所有網卡）
    depends_on:
      postgres:
        condition: service_healthy  # 等 PostgreSQL 健康後才啟動
    healthcheck:
      test: ["CMD", "wget", "..."]  # 每 10 秒檢查服務是否正常
    networks:
      - frontend-net    # 使用者可存取
      - backend-net     # 其他服務可存取
```

**Port 綁定差異：**
- `0.0.0.0:4000:4000` — 任何人都能存取（frontend 服務）
- `127.0.0.1:5432:5432` — 只有本機能存取（data 服務）

**depends_on + healthcheck：** 確保啟動順序。例如 LiteLLM 需要 PostgreSQL 的資料庫，所以必須等 PostgreSQL 完全啟動（通過健康檢查）後才啟動 LiteLLM。

---

### configs/litellm/config.yaml

```yaml
model_list:
  - model_name: "gemma-4-26b"              # 你的應用程式用這個名字呼叫
    litellm_params:
      model: "openai/gemma-4-26b-a4b-it-4bit"  # 實際的模型標識
      api_base: "os.environ/LOCAL_LLM_BASE_URL" # 從環境變數讀取 URL
      api_key: "no-key-needed"                  # 本地推論不需要 Key

router_settings:
  fallbacks:
    - gemma-4-26b: ["gpt-4o", "claude-sonnet"]  # 失敗時依序嘗試

litellm_settings:
  success_callback: ["langfuse"]  # 每次成功呼叫都通知 Langfuse 記錄
```

**`os.environ/XXX` 語法：** LiteLLM 特有的寫法，表示「從環境變數讀取這個值」。這樣敏感資訊（API Key）不用寫死在設定檔中。

---

### configs/postgres/init-databases.sh

```bash
#!/bin/bash
# PostgreSQL 啟動時自動執行此腳本
# 建立多個資料庫給不同服務使用

create_db_and_user() {
    psql -U "$POSTGRES_USER" <<-EOSQL
        CREATE USER $1 WITH PASSWORD '$POSTGRES_PASSWORD';
        CREATE DATABASE $1 OWNER $1;
    EOSQL
}

create_db_and_user litellm    # LiteLLM 用（存 Virtual Key、用量紀錄）
create_db_and_user langfuse   # Langfuse 用（存 Trace 資料）
create_db_and_user mlflow     # MLflow 用（存實驗紀錄）
create_db_and_user dagster    # Dagster 用（存 Pipeline 狀態）
```

**為什麼共用一個 PostgreSQL？** PoC 階段簡化部署。生產環境建議各服務獨立資料庫。

---

### configs/prometheus/prometheus.yml

```yaml
global:
  scrape_interval: 15s  # 每 15 秒抓一次指標

scrape_configs:
  - job_name: litellm
    static_configs:
      - targets: ["litellm:4000"]  # 用 Docker 內部 DNS 名稱
    metrics_path: /metrics         # LiteLLM 暴露指標的路徑
```

**Prometheus 的 Pull 模式：** 不是服務「推」資料給 Prometheus，而是 Prometheus 主動去「拉」。這簡化了服務端的實作。

---

### configs/guardrails/

**config.yml** — 定義使用哪個 LLM 來做安全判斷，以及啟用哪些規則。

**prompts.yml** — 定義安全檢查用的 Prompt。例如讓 LLM 判斷「這段文字是否包含 PII」。

**rails/pii.co** — 用 Colang（NeMo 的領域語言）定義 PII 的範例和處理流程：

```colang
define user contains pii
  "My credit card number is 4111-1111-1111-1111"  ← 範例模式
  "My ID number is A123456789"

define flow pii filter input
  user contains pii     ← 如果偵測到匹配
  bot refuse pii        ← 回覆拒絕訊息
  stop                  ← 停止處理，不送給 LLM
```

---

## 腳本詳解

### agents/react_agent.py — ReAct Agent

**驗證項目：** V-15（多步驟推論）、V-16（Tool Calling）

**原理：** Agent 在一個循環中交替「思考」和「行動」：

```python
# 簡化後的核心邏輯
def agent_node(state):
    """LLM 決定下一步：呼叫工具或直接回答"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state):
    """判斷 LLM 是否要呼叫工具"""
    if last_message.tool_calls:
        return "tool_node"  # 繼續執行工具
    return END              # 直接回答，結束

# 圖結構: agent → (有工具呼叫?) → tool_node → agent → ... → END
```

**三個工具：**
- `calculator`：安全的數學計算（用 allowlist 防止 `eval()` 注入攻擊）
- `search_web`：模擬網路搜尋（PoC 用 mock 資料）
- `get_current_time`：取得當前時間

---

### agents/hitl_agent.py — Human-in-the-Loop Agent

**驗證項目：** V-17

**原理：** 使用 LangGraph 的 `interrupt()` 機制暫停圖的執行：

```python
def human_review_node(state):
    """暫停圖執行，等待外部輸入"""
    decision = interrupt({       # ← 這裡暫停！
        "action_plan": state["action_plan"],
        "message": "請審查以上計劃"
    })
    return {"human_approved": decision == "approve"}
```

**MemorySaver：** 暫停時，整個圖的狀態被持久化到記憶體中。當人類回覆後，可以從斷點恢復。

---

### agents/rag_agent.py — RAG + Guardrails Agent

**驗證項目：** 綜合 V-10、V-12、V-13

**流程：**

```python
# 線性圖：retrieve → generate → guardrails → output
def retrieve_node(state):
    """從 Qdrant 搜尋相關文件"""
    results = qdrant_client.query(...)
    return {"retrieved_context": results}

def generate_node(state):
    """用 LLM + Context 產生回答"""
    prompt = f"根據以下資料回答：{state['retrieved_context']}\n問題：{state['user_question']}"
    answer = llm.invoke(prompt)
    return {"generated_answer": answer}

def guardrails_node(state):
    """用 NeMo Guardrails 檢查回答是否安全"""
    result = httpx.post("http://localhost:8090/v1/chat/completions", ...)
    return {"guardrails_result": result}
```

---

### agents/mem0_agent.py — 記憶管理

**驗證項目：** V-19

**核心操作：**

```python
from mem0 import Memory

config = {
    "vector_store": {"provider": "qdrant", ...},  # 向量存 Qdrant
    "llm": {"provider": "openai", ...},           # 用 LiteLLM 做摘要
}

memory = Memory.from_config(config)

# 儲存記憶
memory.add("我在 TechCorp 當工程師", user_id="alice")

# 搜尋記憶（語意搜尋，不是關鍵字比對）
results = memory.search("Alice 在哪裡工作？", user_id="alice")

# 取得所有記憶
all_memories = memory.get_all(user_id="alice")

# 刪除特定記憶
memory.delete(memory_id="...")
```

---

### rag/rag_pipeline.py — RAG 完整流程

**驗證項目：** V-10、V-11

**五步驟：**

```python
# Step 1: 載入文件
docs = DirectoryLoader("./data/", glob="**/*.txt").load()

# Step 2: 切片（重要！大文件要切成小段）
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # 每段最多 1000 字元
    chunk_overlap=200    # 相鄰段落重疊 200 字元（避免切斷句子）
)
chunks = splitter.split_documents(docs)

# Step 3: 向量化 + 存入 Qdrant
vectorstore = Qdrant.from_documents(chunks, embeddings, ...)

# Step 4: 建立 QA 鏈
chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(base_url="http://localhost:4000/v1"),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),  # 取前 3 篇
)

# Step 5: 問答
answer = chain.invoke({"query": "什麼是 LLMOps？"})
```

**chunk_overlap 的作用：** 如果一個完整的句子剛好被切在兩段之間，overlap 確保兩段都包含這個句子，不會遺失語意。

---

### rag/eval_ragas.py — RAG 品質評估

**驗證項目：** V-09

**四個評估指標：**

| 指標 | 衡量什麼 | 分數意義 |
|------|---------|---------|
| **Faithfulness** | 回答是否忠於檢索到的文件 | 高分 = 沒有「幻覺」 |
| **Answer Relevancy** | 回答是否切題 | 高分 = 回答了問題 |
| **Context Precision** | 檢索的文件是否精確 | 高分 = 沒有無關文件 |
| **Context Recall** | 是否找到所有相關文件 | 高分 = 沒有遺漏 |

```python
# 測試資料結構
{
    "question": "什麼是 RAG？",
    "answer": "RAG 是一種結合檢索和生成的技術...",     # 模型的回答
    "contexts": ["RAG 全名 Retrieval-Augmented..."],  # 檢索到的文件
    "ground_truth": "RAG 是 Retrieval-Augmented..."   # 標準答案
}
```

---

### mlops/mlflow_experiment.py — 實驗追蹤

**驗證項目：** V-20、V-21

```python
with mlflow.start_run():
    # 記錄超參數
    mlflow.log_params({"n_estimators": 100, "max_depth": 5})
    
    # 記錄指標
    mlflow.log_metrics({"accuracy": 0.95, "f1_score": 0.94})
    
    # 記錄模型
    mlflow.sklearn.log_model(model, "model")

# 模型註冊 + 階段升級
client = MlflowClient()
client.create_registered_model("iris-classifier")
client.transition_model_version_stage("iris-classifier", 1, "Production")
```

---

### mlops/dagster_pipeline.py — 管線編排

**驗證項目：** V-22

```python
@asset
def raw_data():
    """資產 1: 載入原始資料"""
    return load_iris_as_dataframe()

@asset
def feature_engineered_data(raw_data):
    """資產 2: 特徵工程（依賴 raw_data）"""
    df = raw_data.copy()
    df["petal_ratio"] = df["petal_length"] / df["petal_width"]
    return df

@asset
def model_evaluation(feature_engineered_data):
    """資產 3: 訓練 + 評估 + 記錄到 MLflow"""
    model = train(feature_engineered_data)
    mlflow.log_metrics(evaluate(model))
    return metrics
```

**Software-Defined Assets：** Dagster 的核心理念 — 你定義「最終想要什麼」（asset），框架幫你管理「怎麼產生」和「何時需要重新產生」。

---

### scripts/seed_qdrant.py — 向量 DB 初始化

**驗證項目：** V-10

這個腳本做三件事：
1. 在 Qdrant 建立一個集合（collection），設定向量維度（768）和距離計算方式（cosine）
2. 用本地 Embedding 模型把 5 篇示範文件轉成向量
3. 測試搜尋功能是否正常

---

### scripts/test_integration.py — 整合測試

**驗證項目：** V-01 ~ V-25（全部）

自動化驗證 25 個項目，輸出類似：

```
╔══════╦══════════════════════════════════════╦════════╗
║ V-01 ║ LiteLLM → Local Gemma 4            ║  PASS  ║
║ V-02 ║ LiteLLM → Azure OpenAI             ║  PASS  ║
║ V-03 ║ LiteLLM Fallback                   ║  PASS  ║
║ ...  ║ ...                                ║  ...   ║
║ V-25 ║ Docker Compose 全棧啟停             ║  PASS  ║
╚══════╩══════════════════════════════════════╩════════╝
Total: 18 PASS / 2 FAIL / 5 SKIP
```

對於需要手動驗證的項目（如 Agent 腳本），會標記為 SKIP 並提示執行方式。

---

## 驗證項目清單

| 編號 | 領域 | 項目 | 對應腳本 / 設定 |
|------|------|------|----------------|
| V-01 | LLMOps | LiteLLM → 本地 Gemma 4 | `configs/litellm/config.yaml` |
| V-02 | LLMOps | LiteLLM → Azure OpenAI | `configs/litellm/config.yaml` |
| V-03 | LLMOps | Fallback 機制 | `configs/litellm/config.yaml` (router_settings) |
| V-04 | LLMOps | Virtual Key 認證 | LiteLLM UI |
| V-05 | LLMOps | 成本追蹤 | LiteLLM /spend API |
| V-06 | LLMOps | Langfuse Trace | `configs/litellm/config.yaml` (success_callback) |
| V-07 | LLMOps | 延遲分析 | Langfuse Dashboard |
| V-08 | LLMOps | Prompt 管理 | Langfuse UI |
| V-09 | LLMOps | RAG 評估 | `rag/eval_ragas.py` |
| V-10 | LLMOps | Qdrant 向量儲存 | `scripts/seed_qdrant.py` |
| V-11 | LLMOps | RAG 端到端 | `rag/rag_pipeline.py` |
| V-12 | LLMOps | PII 過濾 | `configs/guardrails/rails/pii.co` |
| V-13 | LLMOps | 主題限制 | `configs/guardrails/rails/topic.co` |
| V-14 | LLMOps | Chat UI | Open WebUI |
| V-15 | AgentOps | 單 Agent 執行 | `agents/react_agent.py` |
| V-16 | AgentOps | Tool Calling | `agents/react_agent.py` |
| V-17 | AgentOps | Human-in-the-Loop | `agents/hitl_agent.py` |
| V-18 | AgentOps | Agent Trace | `agents/react_agent.py` (Langfuse callback) |
| V-19 | AgentOps | 記憶管理 | `agents/mem0_agent.py` |
| V-20 | MLOps | MLflow 實驗 | `mlops/mlflow_experiment.py` |
| V-21 | MLOps | Model Registry | `mlops/mlflow_experiment.py` |
| V-22 | MLOps | Dagster Pipeline | `mlops/dagster_pipeline.py` |
| V-23 | Infra | Prometheus 指標 | `configs/prometheus/prometheus.yml` |
| V-24 | Infra | Grafana Dashboard | `configs/grafana/dashboards/llmops-overview.json` |
| V-25 | Infra | Docker Compose 啟停 | `docker-compose.yml` |

---

## 常見問題

### Q: 沒有 GPU 可以跑嗎？

可以，但需要修改 `configs/litellm/config.yaml`，把 fallback 中的雲端模型設為 primary。或者使用更小的本地模型（如 Gemma 2B）。

### Q: 為什麼用 Gemma 4 而不是 GPT-4o？

- **資料不離開本地**：企業環境中，資料主權很重要
- **零 API 費用**：只有電費
- **低延遲**：不需要網路往返
- **雲端作為 Fallback**：本地掛了還有備援

### Q: 32GB RAM 能跑嗎？

勉強可以，但只能跑核心子集：LiteLLM + Langfuse v2（去掉 ClickHouse + MinIO） + Qdrant + PostgreSQL + Redis。其他服務需要關閉。

### Q: Docker Compose 啟動後某個服務一直重啟？

```bash
# 查看該服務的日誌
docker compose logs -f <service-name>

# 常見原因：
# - PostgreSQL 還沒準備好（等幾分鐘，healthcheck 會自動重試）
# - .env 中的密碼不匹配
# - Port 被其他程式佔用（如 3000 被 React dev server 佔）
```

### Q: 如何新增一個 LLM Provider？

編輯 `configs/litellm/config.yaml`，在 `model_list` 中新增：

```yaml
  - model_name: "my-new-model"
    litellm_params:
      model: "provider/model-name"
      api_base: "https://api.example.com"
      api_key: "os.environ/MY_NEW_API_KEY"
```

然後在 `.env` 中新增 `MY_NEW_API_KEY=xxx`，重啟 LiteLLM。

### Q: 如何停止所有服務？

```bash
docker compose down          # 停止並移除容器
docker compose down -v       # 同上 + 刪除所有持久化資料（小心！）
```

---

## 授權

本專案為教學目的建立的 PoC，相關工具各有其授權條款。
