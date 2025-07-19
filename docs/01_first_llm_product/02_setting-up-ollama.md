# Setting Up Ollama for Local LLM Deployment

- The project established an on-premise inference pipeline for Meta’s open-source LLaMA 3 model using the Ollama platform.
- A prototype Korean tutoring application was implemented via prompt engineering, demonstrating cost-free, offline LLM utility.
- Cross-platform installation workflows and user-driven refinements validated both usability and performance benchmarks.

## Context and Objectives

### **Background**

- Rise of proprietary LLM API costs prompted investigation of local inference solutions.
- Meta’s LLaMA series offers competitive open-source weights enabling self-hosted models.

### **Primary Objectives**

- Deploy LLaMA 3 locally on both Windows and macOS with minimal dependencies.
- Validate interactive tutor use case for language instruction without external service fees.
- Document end-to-end workflow for developer adoption.

### **Scope and Constraints**

- **Platforms**: Windows 10+ (PowerShell) and macOS (bash/zsh).
- **Model**: LLaMA 3, using the 8B variant. No official 2B model is available from Meta.
- **Hardware**: CPU inference only; GPU support deferred to future phases.
- **Network**: Initial download required; subsequent runs fully offline.

## Technical Deep Dive

### Architectural Overview

- **Ollama CLI**: C++ executable providing model management, download, and inference.
- **Model Variant**: `llama3:8b`, optimized for CPU performance.
- **Runtime Environment**:

  - Windows: PowerShell with manual `PATH` configuration to include Ollama binary, e.g.:

    ```powershell
    $env:Path += ";C:\Program Files\Ollama"
    ```

  - macOS: Homebrew installation or direct binary invocation in terminal.

- **Storage**: Model weights cached under `~/.ollama/models/blobs`; reused across sessions via internal Ollama indexing.

### Dependency Management

- **Windows Prerequisites**: No external dependencies; native binary installer (no .NET required).
- **macOS Prerequisites**: Standard Xcode CLI tools; optional Homebrew.
- **Networking**: HTTPS endpoint for weight download; retry logic built into Ollama downloader.

### Inference Workflow

```bash
# Windows PowerShell
ollama run llama3:8b

# macOS Terminal
ollama run llama3:8b
```

- On first invocation, progress bar displays byte-level download status.
- Subsequent invocations initiate immediate interactive REPL.

### Prompt Engineering Strategy

- **Initial Prompt**:

  ```sh
  I am trying to learn Korean. 
  I am a complete beginner. 
  Please chat with me in basic Korean to teach me.
  ```

- **Iterative Refinements**:

  - Enforced Korean punctuation and structural formatting.
  - Specified response format: greeting, example phrase, correction hints.

### Challenges and Mitigations

- **Download Latency**: Users reported variable network speeds; resolved by exposing download progress and allowing resume on failure.
- **Cross-Platform Differences**: Path and execution semantics documented in separate OS-specific READMEs.
- **Prompt Sensitivity**: Minor typographic errors led to misformatted output; mitigated via explicit punctuation rules in prompt.

## Lessons Learned and Best Practices

- **Local Inference Viability**: On-premise LLMs eliminate per-token charges and improve data privacy.
- **Model Selection Tradeoffs**: Smaller models offer faster CPU inference but may underperform on complex tasks.
- **User-Centered Prompt Design**: Clearly specify role, proficiency, and expected output format to reduce ambiguity.
- **Documentation Granularity**: Separate OS-specific sections prevent cross-platform confusion.
- **Resumable Downloads**: Built-in retry and progress reporting enhance user experience on unstable networks.
- **Offline Capability**: Caching of model weights ensures functionality without persistent connectivity.

## Recommendations and Next Steps

- **GPU Acceleration Enablement**

  - Integrate CUDA (Linux/Windows) and Metal (macOS) backends for multi-fold speedups.
  - Benchmark CPU vs GPU performance to quantify gains.

- **Fine-Tuning Pipeline Development**

  - Establish data ingestion and preprocessing workflows.
  - Implement LoRA or full-parameter fine-tuning on curated Korean tutoring dataset.

## User Interface Enhancement

- Build minimal web frontend (React) to manage sessions, store conversation history, and visualize progress metrics.

## Automated Evaluation Framework

- Define evaluation metrics (BLEU, perplexity) for language tutoring quality.
- Implement batch testing harness for regression monitoring.

## Scalability and Load Testing

- Simulate concurrent users on shared hardware.
- Instrument resource usage (CPU, memory, disk) and optimize binary flags.

## Appendices

### Glossary

- **Ollama**: Command-line platform for local LLM hosting.
- **LLaMA 3**: Meta’s open-source LLM; this project uses the 8B variant.
- **REPL**: Read-Eval-Print Loop for interactive model sessions.
- **LoRA**: Low-Rank Adaptation, parameter-efficient fine-tuning method.

### Referenced Tools and Frameworks

- **PowerShell** (Windows)
- **bash/zsh** (macOS)
- **Homebrew** (macOS package manager)
- **CUDA Toolkit** (future GPU support)
- **Metal Performance Shaders** (future macOS GPU support)

### Code Snippet: Sample Prompt with Format Enforcement

```bash
ollama run llama3:latest "
You are a Korean tutor.
User level: Beginner.
Respond with:
1. Greeting in Korean.
2. One example phrase.
3. Correction hints for user input.
"
```
