# DataPilot

### Agentic AI-Powered Data Science Assistant

DataPilot is an **agentic AI assistant** that helps **data scientists, data analysts, machine learning engineers, and AI engineers** explore, validate, clean, and manipulate datasets through natural language.

Instead of manually inspecting unfamiliar directory structures or writing one-off Python scripts for every new dataset, users simply describe what they want. The assistant plans the required steps, selects the appropriate tools, explains its reasoning, requests approval for potentially destructive operations, and executes the workflow safely.

Built using **LangChain** and **LangGraph**, DataPilot is an agentic AI assistant that uses tool calling, stateful memory, sandboxed Python execution, and human-in-the-loop verification to automate dataset exploration, analysis, visualization, and manipulation through natural language while safely preserving the original dataset.

---

# 🚀 Highlights

* 🤖 Agentic AI powered by **LangChain**
* 🔧 Tool-calling architecture for autonomous workflow execution
* 🧠 Stateful conversational memory
* 🛡️ Human-in-the-loop execution for safe dataset modifications
* 🐍 Sandboxed Python execution for custom analysis and automation
* 🌐 Multi-provider LLM support

  * OpenAI
  * Google Gemini
  * AWS Bedrock
* 📊 Automatic dataset profiling and validation
* 🧩 Modular and extensible tool framework

---

# 🎯 What Problem Does It Solve?

Data preparation is one of the most time-consuming stages of any data science or machine learning project.

Before building models, engineers often spend significant time:

* Understanding unfamiliar dataset structures
* Calculating train/validation/test distributions
* Identifying annotation formats
* Detecting missing files and inconsistencies
* Writing repetitive Python scripts for cleaning and restructuring
* Generating quick statistics and visualizations

These tasks are repetitive, error-prone, and rarely contribute directly to solving the core machine learning problem.

DataPilot automates these workflows through a conversational interface, allowing users to focus on data analysis and model development instead of repetitive dataset engineering.

---

# ✨ Features

### 📁 Safe Workspace Isolation

Automatically creates a dedicated workspace copy of the dataset before performing any operations.

All analysis and modifications occur on the copied dataset, ensuring the original data is never modified.

---

### 🔍 Automatic Dataset Profiling

Automatically identifies:

* Dataset structure
* Train / validation / test splits
* Split percentages
* Annotation formats
* File counts
* Dataset statistics

without requiring manual inspection.

---

### 📊 Dataset Validation

Detects common dataset issues such as:

* Missing labels
* Orphaned images
* Missing dataset splits
* Structural inconsistencies
* Annotation irregularities

---

### 📈 Dynamic Dataset Visualization & Manipulation

Generates Python scripts on demand to perform:

* Dataset statistics
* Custom analysis
* Visualizations
* Charts
* File inspections
* Dataset transformations

inside an isolated sandboxed runtime.

---

### Human-in-the-Loop Safeguards

All generated Python code follows a strict workflow:

**Propose → Confirm → Execute**

Before running any generated code, the assistant:

1. Explains exactly what will be executed.
2. Describes the expected output or modification.
3. Requests explicit user confirmation. (if required)

Potentially destructive operations (deletions, renaming, restructuring) always require approval before execution.

---

### 🌐 Multi-LLM Support

Supports multiple providers through a unified interface:

* OpenAI
* Google Gemini
* AWS Bedrock

Switching providers only requires updating the `.env` configuration.

---

# 📋 Example Tasks

The assistant can:

* Explore unfamiliar datasets
* Detect dataset structures
* Calculate split distributions
* Identify annotation formats
* Visualize class distributions
* Generate dataset statistics
* Detect orphaned images or labels
* Find structural inconsistencies
* Convert annotation formats
* Rename files
* Reorganize directory structures
* Clean datasets
* Execute custom Python analyses
* Generate charts and visualizations

---

# 🏗 System Architecture

DataPilot uses LangChain's agent framework together with LangGraph's checkpointing and memory components to provide tool orchestration and stateful conversations.

```
                           User
                             │
                             ▼
                  Natural Language Request
                             │
                             ▼
                ┌────────────────────────┐
                │    LangChain Agent     │
                │  (Planning & Reasoning)│
                └───────────┬────────────┘
                            │
                  Select appropriate tool
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
 ┌─────────────────────┐               ┌──────────────────────┐
 │ Dataset Tooling     │               │ Python Runtime       │
 │                     │               │                      │
 │ • copy_dataset()    │               │ • Analysis           │
 │ • explore_structure()│              │ • Visualizations     │
 │ • get_split_summary()│              │ • Data Cleaning      │
 │ • inspect_annotation│               │ • Transformations    │
 │ • ask_user()        │               │                      │
 └──────────┬──────────┘               └──────────┬───────────┘
            │                                     │
            └───────────────────┬─────────────────┘
                                ▼
                    Human Confirmation (if required)
                                │
                                ▼
                     Execute Tool / Python Script
                                │
                                ▼
                      Update Conversation State
                                │
                                ▼
            Continue Agent Reasoning by LangChain Agent                                        
```

---

# 🧩 Core Tooling

The assistant currently provides the following tools:

| Tool                          | Description                                                 |
| ----------------------------- | ----------------------------------------------------------- |
| `copy_dataset()`              | Creates a safe workspace copy of the dataset                |
| `explore_structure()`         | Scans and caches the dataset directory tree                 |
| `get_split_summary()`         | Calculates dataset split statistics                         |
| `inspect_annotation_format()` | Detects annotation formats                                  |
| `run_python()`                | Executes generated Python scripts in an isolated subprocess |
| `ask_user()`                  | Requests clarification or execution approval                |

---

# 💻 Installation

## 1. Clone the repository

```bash
git clone https://github.com/r-muthu/datapilot.git
cd datapilot
```

---

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Configure your model provider

Create a `.env` file.

### OpenAI

```env
MODEL_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-5
```

---

### Google Gemini

```env
MODEL_PROVIDER=gemini
GOOGLE_API_KEY=your_api_key
GEMINI_MODEL=gemini-2.5-flash
```

---

### AWS Bedrock

```env
MODEL_PROVIDER=bedrock
AWS_BEARER_TOKEN_BEDROCK=your_bedrock_bearer_token
BEDROCK_MODEL=openai.gpt-oss-safeguard-120b
AWS_REGION=us-east-1
```

Configure AWS credentials using the standard AWS credential chain (`aws configure`, IAM roles, or environment variables).

---

> **Note**
>
> When experimenting with alternative or fine-tuned models, ensure they support **Tool Calling / Function Calling**. Models without reliable tool-calling capabilities may repeatedly invoke the same tool, fail to terminate correctly, or produce significantly slower responses.

---

# ▶ Running the Assistant

```bash
python agent.py
```

---

# 💬 Example Session

```text
=== DataPilot ===

Enter the full path to your dataset (use /mnt/c/... for WSL): /home/user/datasets/vehicles_v1

[Agent]: Workspace copy created successfully at
`/home/user/datasets/vehicles_v1_copy_20260702_174000`.

I have profiled the dataset:

- Dataset Type: Computer Vision (Images + Labels)
- Total Files: 2,506 files
- Splits:
    * train: 878 images, 878 labels (70.1%)
    * valid: 250 images, 250 labels (19.9%)
    * test: 126 images, 126 labels (10.0%)
- Annotation Format: YOLO text configuration format (`.txt`). Sample inspected: `train/labels/001.txt`

**What would you like to do next?**

- **Explore further** (e.g., inspect specific images/XML files, check other statistics).
- **Manipulate data** (e.g., rename classes, merge splits, clean mismatches, etc.).
- **Finish** – end the session.

Your request (or 'done'): Find out if there are any orphaned images without labels, and state them.

[Agent]: I will write a Python script that aggregates all file stems in the image directories and cross-references them against the label directories to look for unmatched files and return their names.

Would you like me to execute this script to find mismatches? (yes/no): yes

[Agent]: Analysis complete!

Found 3 images in `train/images` with no corresponding label files:

- `img1.jpg`
- `img2.jpg`
- `img3.jpg`

Found 3 orphaned label files with no corresponding images:

- `img01.txt`
- `img02.txt`
- `img03.txt`

**What would you like to do next?**

- **Explore further** (e.g., inspect specific images/XML files, check other statistics).
- **Manipulate data** (e.g., rename classes, merge splits, clean mismatches, etc.).
- **Finish** – end the session.

Your request (or 'done'): Generate class distributions and bar charts for each split.

[Agent]: I will write a Python script that counts the occurrences of each class ID in every dataset split and generates bar charts showing the class distribution for the training, validation and test sets.

Would you like me to execute this script to generate the statistics and charts? (yes/no): yes

[Agent]: Analysis complete!

Class distributions:

| Split | Class ID → Count |
|-------|------------------|
| **train** | 0: 170, 1: 198, 2: 914, 3: 202, 4: 192 |
| **valid** | 0: 64, 1: 46, 2: 238, 3: 46, 4: 60 |
| **test** | 0: 18, 1: 38, 2: 150, 3: 32, 4: 20 |

Generated bar chart images:

- `train_class_distribution.png`
- `valid_class_distribution.png`
- `test_class_distribution.png`

The images have been saved to the workspace directory.

**What would you like to do next?**

- **Explore further** (e.g., inspect specific images/XML files, check other statistics).
- **Manipulate data** (e.g., rename classes, merge splits, clean mismatches, etc.).
- **Finish** – end the session.

Your request (or 'done'): done
```

---

# 🔒 Safety

* Original datasets are never modified.
* Every operation runs inside the workspace copy unless explixitly stated in the 'enter full path to dataset' prompt.
* Generated Python executes inside an isolated subprocess.
* Destructive operations always require explicit approval.
* Failed Python executions are automatically repaired and retried (up to three attempts) before requesting user intervention.

---

# 🧠 Engineering Concepts Demonstrated

This project showcases:

* Agentic AI
* LangChain Agents
* Tool Calling / Function Calling
* ReAct-style reasoning
* Human-in-the-loop AI
* Multi-provider LLM abstraction
* Sandboxed code execution
* Prompt engineering
* Modular software architecture
* Error recovery and retry strategies

---

# 🛣 Roadmap

Future enhancements include:

* SQL database exploration
* Statistical profiling
* Data quality reporting
* Feature engineering tools
* Automated report generation
* Dataset versioning
* Model evaluation workflows
* RAG over datasets
* Multi-agent collaboration
* Support for NLP and tabular datasets

---