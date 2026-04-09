# 🌍 Multi-Agent Travel Planner (AI 智能旅行规划管家)

![Python Version](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)
![Architecture](https://img.shields.io/badge/Architecture-Multi--Agent-success.svg)

基于大语言模型 (LLM) 与 Multi-Agent 架构构建的智能旅行规划系统。系统通过编排多个垂直领域的 AI Agent，结合 RAG（检索增强生成）与真实 LBS（高德地图）数据，旨在解决传统 AI 生成行程时产生的“幻觉”问题，为用户提供精确、可执行、甚至带有预算控制的端到端旅行方案。

## ✨ 核心架构与亮点 (Core Features)

* **🤖 多智能体协同 (Multi-Agent Orchestration):**
  系统解耦了 7 个具备独立 Prompt 与工具链的 Agent（偏好分析、目的地规划、航班、酒店、餐饮、活动、预算管理），支持串行与并行任务调度，大幅提升规划效率与专业度。
* **🧠 记忆与 RAG 机制 (Memory & RAG):**
  引入本地向量数据库 (ChromaDB) 持久化用户偏好与历史对话，确保长程对话中的上下文连贯性，实现“越用越懂你”的个性化体验。
* **📍 真实物理世界锚定 (Anti-Hallucination):**
  深度接入高德地图 LBS API 与 Tavily 实时搜索。从“赛博朋克式”的虚构行程，升维到基于真实地理位置、营业时间、真实路线规划的落地级方案。
* **💰 动态预算回旋机制 (Budget Loop Controller):**
  设计了超预算重试与降级策略。当规划总价超出用户预算时，预算 Agent 会自动触发重规划流程，动态调整酒店星级或交通方式。

## 🛠️ 技术栈 (Tech Stack)

* **核心语言:** Python 3.11+
* **LLM 引擎:** DeepSeek / OpenAI 兼容接口
* **Agent 框架:** 自研轻量级 Agent 编排逻辑 / LangChain
* **向量数据库:** ChromaDB (本地持久化)
* **外部工具 (Tools):** 高德开放平台 Web 服务 API, Tavily Search API
* **前端展示:** Streamlit

## 📂 项目结构 (Project Structure)

```text
├── python/
│   ├── agents/          # 核心 Agent 定义 (Flight, Hotel, Budget 等)
│   ├── api/             # 外部 API 接口封装
│   ├── config/          # 全局配置与环境变量加载
│   ├── models/          # Pydantic 数据模型定义
│   ├── orchestrator/    # 多 Agent 调度器 (并行/串行控制)
│   ├── tools/           # Agent 可调用的具体工具 (高德 API、搜索等)
│   ├── ui/              # Streamlit 界面交互
│   └── utils/           # 通用工具类
├── .env.example         # 环境变量配置模板
├── requirements.txt     # Python 依赖清单
└── plan.md              # 架构设计与演进文档
```

## 🚀 快速启动 (Quick Start)

### 1. 克隆项目
```bash
git clone [https://github.com/zsd1024/multi-agent-travel-planner.git](https://github.com/zsd1024/multi-agent-travel-planner.git)
cd multi-agent-travel-planner/python
```

### 2. 环境配置
建议使用 Python 虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

pip install -r ../requirements.txt
```

### 3. 配置密钥 (API Keys)
复制环境变量模板并填入你的专属 Key：
```bash
cp .env.example .env
```
在 `.env` 文件中配置以下内容：
* `LLM_API_KEY`: 你的大模型 API 密钥
* `AMAP_API_KEY`: 高德开放平台 Web 服务 Key
* `TAVILY_API_KEY`: Tavily Search Key

### 4. 启动服务
```bash
streamlit run ui/streamlit_app.py
```

运行示例：
<img width="2557" height="1244" alt="image" src="https://github.com/user-attachments/assets/dd035a5e-9da1-4833-b32c-0d382d305031" />


## 🗺️ 未来演进方向 (TODO)

- [ ] 接入真实航班/酒店预订系统 (OTA 平台 API)
- [ ] 支持多轮对话中的局部行程修改 (如：“把第二天的行程换成室内活动”)
- [ ] 导出行程为 PDF 或日历格式 (.ics)

---
*Developed by zsd1024. Welcome to fork and contribute!*
