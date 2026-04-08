# Forge AI — MLOps Agent

**LLM-powered assistant for the full ML workflow: data → model → evaluation**

An intelligent agent built with LangChain that assists ML practitioners through the entire machine learning lifecycle. Uses RAG over ML/DL documentation, tool-calling for dataset analysis and model evaluation, and structured prompt engineering — all powered by free-tier LLM APIs (Groq, Gemini) via the OpenAI-compatible SDK.

[![Python](https://img.shields.io/badge/Python-3.10%2B-2ea44f?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Agents-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![OpenAI SDK](https://img.shields.io/badge/OpenAI_SDK-Compatible-412991?style=for-the-badge&logo=openai&logoColor=white)](https://platform.openai.com/docs)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-FF6F00?style=for-the-badge)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

---

## What is Forge AI?

Forge is an **LLM agent that understands ML workflows**. Instead of switching between docs, Stack Overflow, and notebooks, you talk to Forge:

```
You: I have a CSV with 50k rows and 30 features, some with missing values.
     I want to predict customer churn. What's the best approach?

Forge: [🔧 analyze_dataset] → Inspecting your data...
       [📚 rag_query] → Retrieving relevant sklearn documentation...

       Your dataset has 12% missing values concentrated in 4 columns,
       moderate class imbalance (18% churn), and 8 categorical features.

       Recommended approach:
       1. Imputation: IterativeImputer for the 4 columns with missing data
       2. Encoding: TargetEncoder for high-cardinality categoricals
       3. Model: Start with GradientBoostingClassifier (handles imbalance well)
       4. Evaluation: Use PR-AUC over ROC-AUC given the imbalance

       Here's the starter pipeline:
       [🔨 generate_code] → Generating scikit-learn pipeline...
```

---

## Architecture

```
flowchart LR
  U[User Query] --> A[LangChain Agent]
  A --> R{Route}
  R --> T1[🔧 analyze_dataset<br/>Profile & EDA]
  R --> T2[📚 rag_query<br/>ML Docs RAG]
  R --> T3[🔨 generate_code<br/>Pipeline Boilerplate]
  R --> T4[📊 evaluate_model<br/>Metrics & Diagnostics]
  R --> T5[🧹 suggest_preprocessing<br/>Data Cleaning]
  R --> T6[🔍 explain_concept<br/>ML Concepts]
  T1 & T2 & T3 & T4 & T5 & T6 --> S[Synthesized Response]
```

```
forge-ai/
├── src/
│   ├── agent/                  # LangChain agent core
│   │   ├── __init__.py
│   │   ├── forge.py            # Main agent orchestrator
│   │   ├── llm.py              # LLM client (OpenAI SDK — Groq/Gemini)
│   │   └── memory.py           # Conversation memory management
│   ├── api/                    # FastAPI Backend
│   │   ├── __init__.py
│   │   └── main.py             # REST API for the agent
│   ├── tools/                  # Agent tools (LangChain Tools)
│   │   ├── __init__.py
│   │   ├── dataset_analyzer.py # CSV/dataframe profiling & EDA
│   │   ├── code_generator.py   # ML pipeline boilerplate generation
│   │   ├── model_evaluator.py  # Metrics, confusion matrix, diagnostics
│   │   ├── preprocessing.py    # Data cleaning & feature engineering suggestions
│   │   └── concept_explainer.py# ML concept explanations with examples
│   ├── rag/                    # RAG pipeline over ML documentation
│   │   ├── __init__.py
│   │   ├── ingest.py           # Chunk & embed docs into ChromaDB
│   │   ├── retriever.py        # Semantic retrieval with reranking
│   │   └── sources.py          # Documentation source definitions
│   ├── prompts/                # Prompt engineering templates
│   │   ├── __init__.py
│   │   ├── system.py           # System prompts (agent persona, rules)
│   │   ├── few_shot.py         # Few-shot examples per tool
│   │   └── templates.py        # Task-specific prompt templates
│   └── app.py                  # Streamlit chat interface
├── configs/
│   ├── llm.yaml                # LLM provider configuration
│   ├── rag.yaml                # RAG settings (chunk size, top_k, etc.)
│   └── tools.yaml              # Tool-specific settings
├── data/
│   ├── docs/                   # ML documentation for RAG
│   ├── vectorstore/            # ChromaDB persistence
│   └── sample_datasets/        # Example CSVs for demos
├── tests/
│   ├── test_agent.py
│   ├── test_tools.py
│   └── test_rag.py
├── scripts/
│   ├── ingest_docs.py          # Download & ingest ML docs
│   └── demo.py                 # CLI demo script
├── docs/
│   ├── PROMPT_ENGINEERING.md   # Prompt design decisions & strategies
│   └── TOOLS.md                # Tool documentation & examples
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Features

### 🔧 Tools (Agent Capabilities)

| Tool | What it does | Example |
|------|-------------|---------|
| `analyze_dataset` | Profiles a CSV: shape, dtypes, missing values, distributions, correlations | "Analyze my dataset and tell me what stands out" |
| `rag_query` | Retrieves relevant ML/DL documentation from indexed sources | "How does IterativeImputer handle categorical features?" |
| `generate_code` | Generates scikit-learn/PyTorch pipeline boilerplate | "Generate a training pipeline for binary classification" |
| `evaluate_model` | Computes metrics, plots confusion matrix, suggests improvements | "Evaluate my model's predictions — here are y_true and y_pred" |
| `suggest_preprocessing` | Analyzes data issues and recommends cleaning strategies | "My data has 30% nulls in 3 columns, what should I do?" |
| `explain_concept` | Explains ML concepts with examples and analogies | "Explain cross-validation like I'm new to ML" |
| `track_experiment` | Logs hyperparameters and metrics to a local MLflow server | "Save my run 'baseline_rf' with n_estimators=100 and acc=0.85" |

### 📚 RAG Knowledge Base

Forge has indexed documentation from:
- **scikit-learn** — Preprocessing, models, metrics, pipelines
- **PyTorch** — Modules, training loops, data loading
- **pandas** — Data manipulation, cleaning, analysis
- **Common ML patterns** — Best practices, anti-patterns, debugging guides

### 🎯 Prompt Engineering

Documented prompt strategies in [`docs/PROMPT_ENGINEERING.md`](docs/PROMPT_ENGINEERING.md):
- **System prompt design**: Agent persona with ML expertise constraints
- **Few-shot examples**: Per-tool demonstrations for consistent output
- **Chain-of-thought**: Step-by-step reasoning for complex ML decisions
- **Output formatting**: Structured responses with code blocks and explanations
│   ├── tools/                  # Agent tools (LangChain Tools)
# ... rest of the tree ...

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/PaulinaIA/forge-ai.git
cd forge-ai
pip install -e ".[dev]"
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env — choose ONE provider (all free, no credit card):
# GROQ_API_KEY=...     → Llama 3.3 70B (recommended, fastest)
# GEMINI_API_KEY=...   → Gemini 2.0 Flash
```

### 3. Ingest ML Documentation (RAG)

```bash
python scripts/ingest_docs.py
```

### 4. Run the Application

The application is split into a REST API backend and a Streamlit frontend.

**Option A: Using Docker Compose**
```bash
docker-compose up --build
# API runs on http://localhost:8000
# UI runs on http://localhost:8501
```

**Option B: Running locally**
Terminal 1 (Backend API):
```bash
uvicorn forge_ai.api.main:app --reload
```

Terminal 2 (Frontend UI):
```bash
streamlit run app.py
```

# Python API Usage (direct instantiation)
```python
from forge_ai import ForgeAgent
agent = ForgeAgent()
response = agent.run("How should I handle class imbalance in my dataset?")
```

---

## How It Works

### 1. LLM Client (OpenAI-Compatible SDK)

All LLM calls go through the OpenAI SDK, making it trivial to swap providers:

```python
from openai import OpenAI

# Groq (free — Llama 3.3 70B)
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Same code, different provider — no changes needed
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": query}]
)
```

### 2. LangChain Agent with Tools

The agent decides which tools to use based on the query:

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool

@tool
def analyze_dataset(file_path: str) -> str:
    """Profile a CSV dataset: shape, types, missing values, distributions."""
    df = pd.read_csv(file_path)
    # ... profiling logic
    return profile_report

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, memory=memory)
result = executor.invoke({"input": "Analyze my training data"})
```

### 3. RAG over ML Documentation

```python
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Retrieve relevant docs
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Augmented generation
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)
result = qa_chain.invoke({"query": "How to use ColumnTransformer with Pipeline?"})
```

### 4. Prompt Engineering

```python
SYSTEM_PROMPT = """You are Forge, an ML Engineering assistant.

## Your Expertise
- Data preprocessing & feature engineering
- Model selection, training, and evaluation
- scikit-learn pipelines, PyTorch training loops
- MLOps best practices

## Rules
- Always explain WHY, not just WHAT
- Suggest the simplest approach first, then alternatives
- Include code snippets with inline comments
- When uncertain, say so and suggest how to investigate
- Use chain-of-thought: think step by step before recommending

## Output Format
1. Brief assessment of the problem
2. Recommended approach with reasoning
3. Code implementation (if applicable)
4. Next steps & potential pitfalls"""
```

---

## Example Interactions

### Data Exploration
```
You: I uploaded sales_data.csv. What can you tell me about it?

Forge: [🔧 analyze_dataset] Profiling your dataset...

       📊 Dataset Overview:
       • Shape: 45,231 rows × 18 columns
       • Missing: 2,104 values (4.6%) in 3 columns
       • Target: 'revenue' (continuous, right-skewed)
       • Issues: 'date' column parsed as string, 2 near-constant features

       Recommendations:
       1. Parse 'date' to datetime → extract month, day_of_week, is_weekend
       2. Drop 'store_id_legacy' and 'flag_deprecated' (near-zero variance)
       3. Impute 'region_code' with mode (only 1.2% missing)
       4. Consider log-transform on 'revenue' to reduce skewness
```

### Model Selection
```
You: Which model should I use for tabular classification with 50 features?

Forge: [📚 rag_query] Searching ML documentation...

       For tabular data with 50 features, here's my recommendation path:

       Start here → GradientBoostingClassifier or HistGradientBoosting
       • Why: Best out-of-the-box for tabular data, handles mixed types
       • Baseline in 5 min: no feature scaling needed

       If you need speed → RandomForestClassifier
       • Why: Parallelizable, less prone to overfitting than single GB

       If you need interpretability → LogisticRegression with SelectKBest
       • Why: Feature coefficients are directly interpretable

       ⚠️ Skip deep learning unless you have >100k samples.
       Research consistently shows gradient boosting wins on tabular.
```

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM | Groq (Llama 3.3 70B) / Gemini | Free, fast, OpenAI-compatible |
| Agent Framework | LangChain | Tool calling, memory, chains |
| RAG | ChromaDB + LangChain | Lightweight, local, no infra needed |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) | Free, local, no API needed |
| Data Analysis | pandas, numpy, scipy | Dataset profiling tools |
| UI | Streamlit | Fast prototyping, chat interface |
| Containerization | Docker | Reproducible deployment |

---

## Prompt Engineering Documentation

See [`docs/PROMPT_ENGINEERING.md`](docs/PROMPT_ENGINEERING.md) for detailed documentation on:

- **System prompt design**: How the agent persona was crafted and iterated
- **Few-shot strategy**: Examples per tool with input/output pairs
- **Chain-of-thought prompting**: Step-by-step reasoning for complex decisions
- **Temperature & sampling**: Settings per task type (code gen vs explanation)
- **Prompt evaluation**: How prompts were tested and refined
- **Failure modes**: Known edge cases and how prompts handle them

---

## Roadmap

- [x] Core agent with LangChain tool calling
- [x] RAG over scikit-learn / PyTorch docs
- [x] Dataset analysis tool
- [x] Code generation tool
- [x] Model evaluation tool
- [x] Streamlit chat UI
- [ ] Experiment tracking integration (MLflow)
- [ ] Auto-generate full training scripts (not just snippets)
- [ ] Multi-file project scaffolding
- [ ] Fine-tuning advisor tool

---

## Contact

📩 [pauliperalta97@gmail.com](mailto:pauliperalta97@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/paulina-peralta-916a46140/)
🐙 [GitHub](https://github.com/PaulinaIA)
