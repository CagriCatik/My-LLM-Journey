# Large Langugage Model and Agents

This guide presents a comprehensive, step-by-step roadmap for mastering large language models (LLMs) and agentic AI systems. It begins with hands-on projects that take you from zero to a working LLM product, covering environment setup, API integration, prompt design, and foundational applications such as chatbots and retrieval-augmented generation. You will then explore multimodal interfaces, combining text, images, and audio to build richer conversational agents.

Next, you’ll dive into open-source ecosystems-Hugging Face, Google Colab, and community models-learning how to deploy, fine-tune, and benchmark both proprietary and freely available LLMs. Rigorous evaluation methodologies and business-focused benchmarks will sharpen your ability to select and optimize models for real-world code and enterprise tasks. A dedicated section on retrieval-augmented generation with LangChain will teach you how to integrate external knowledge stores and design scalable RAG pipelines.

Fine-tuning techniques using LoRA and QLoRA will equip you with parameter-efficient training skills, culminating in a competitive price-prediction model. The final module extends your expertise to autonomous multi-agent architectures, agentic workflows, and serverless deployment, enabling you to build intelligent systems that operate with minimal human intervention. Each chapter blends theory, practical tutorials, and project assignments to ensure you acquire both the conceptual understanding and the hands-on experience needed to become a proficient LLM engineer.

## Table of Contents

- [Large Langugage Model and Agents](#large-langugage-model-and-agents)
  - [Table of Contents](#table-of-contents)
  - [1: Build Your First LLM Product](#1-build-your-first-llm-product)
  - [2: Build a Multi-Modal Chatbot](#2-build-a-multi-modal-chatbot)
  - [3: Open-Source Gen AI Solutions](#3-open-source-gen-ai-solutions)
  - [4: Model Evaluation for Code \& Business Tasks](#4-model-evaluation-for-code--business-tasks)
  - [5: RAG with LangChain](#5-rag-with-langchain)
  - [6: Fine-Tuning with LoRA/QLoRA](#6-fine-tuning-with-loraqlora)
  - [7: Fine-Tuned Price Prediction Model](#7-fine-tuned-price-prediction-model)
  - [8: Autonomous Multi-Agent Systems](#8-autonomous-multi-agent-systems)
  - [Source](#source)

## 1: Build Your First LLM Product

This first module guides you through every step needed to launch a working LLM application from scratch. You will install and configure local inference tools, deploy models with Ollama on Windows and Mac, and build a practical Spanish‐tutoring assistant to see results immediately. Along the way, you’ll establish a robust development environment-covering Conda, virtualenv, API key management, and best practices for Jupyter Lab-so you can iterate rapidly. By comparing leading models (OpenAI, Ollama, Claude, Gemini) and completing hands-on projects like a web-page summarizer, you’ll acquire the core skills and insights required to progress from beginner to confident LLM engineer.

<details>
<summary>☰ Module Content</summary>

- [ ] Cold Open: Jumping Right into LLM Engineering  
- [ ] Setting Up Ollama for Local LLM Deployment on Windows and Mac  
- [ ] Unleashing the Power of Local LLMs: Build Spanish Tutor Using Ollama  
- [ ] LLM Engineering Roadmap: From Beginner to Master
- [ ] Building LLM Applications: Chatbots, RAG, and Agentic AI Projects  
- [ ] From Wall Street to AI: Ed Donner's Path to Becoming an LLM Engineer  
- [ ] Setting Up Your LLM Development Environment: Tools and Best Practices  
- [ ] Mac Setup Guide: Jupyter Lab and Conda for LLM Projects  
- [ ] Setting Up Anaconda for LLM Engineering: Windows Installation Guide  
- [ ] Alternative Python Setup for LLM Projects: Virtualenv vs. Anaconda Guide  
- [ ] Setting Up OpenAI API for LLM Development: Keys, Pricing & Best Practices  
- [ ] Creating a .env File for Storing API Keys Safely  
- [ ] Instant Gratification Project: Creating an AI-Powered Web Page Summarizer  
- [ ] Implementing Text Summarization Using OpenAI's GPT-4 and Beautiful Soup  
- [ ] Wrapping Up: Key Takeaways and Next Steps in LLM Engineering  
- [ ] LLM Engineering: Key Skills and Tools for AI Development  
- [ ] Understanding Frontier Models: GPT, Claude, and Open Source LLMs  
- [ ] How to Use Ollama for Local LLM Inference: Python Tutorial with Jupyter  
- [ ] Hands-On LLM Task: Comparing OpenAI and Ollama for Text Summarization  
- [ ] Frontier AI Models: Comparing GPT-4, Claude, Gemini, and LLAMA  
- [ ] Comparing Leading LLMs: Strengths and Business Applications  
- [ ] Exploring GPT-4o vs O1 Preview: Key Differences in Performance  
- [ ] Creativity and Coding: Leveraging GPT-4o’s Canvas Feature  
- [ ] Claude 3.5’s Alignment and Artifact Creation: A Deep Dive  
- [ ] AI Model Comparison: Gemini vs Cohere for Whimsical and Analytical Tasks  
- [ ] Evaluating Meta AI and Perplexity: Nuances of Model Outputs  
- [ ] LLM Leadership Challenge: Evaluating AI Models Through Creative Prompts  
- [ ] Revealing the Leadership Winner: A Fun LLM Challenge  
- [ ] Exploring the Journey of AI: From Early Models to Transformers  
- [ ] Understanding LLM Parameters: From GPT-1 to Trillion-Weight Models  
- [ ] GPT Tokenization Explained: How Large Language Models Process Text Input  
- [ ] How Context Windows Impact AI Language Models: Token Limits Explained  
- [ ] Navigating AI Model Costs: API Pricing vs. Chat Interface Subscriptions  
- [ ] Comparing LLM Context Windows: GPT-4 vs Claude vs Gemini 1.5 Flash  
- [ ] Wrapping Up: Key Takeaways and Practical Insights  
- [ ] Building AI-Powered Marketing Brochures with OpenAI API and Python  
- [ ] JupyterLab Tutorial: Web Scraping for AI-Powered Company Brochures  
- [ ] Structured Outputs in LLMs: Optimizing JSON Responses for AI Projects  
- [ ] Creating and Formatting Responses for Brochure Content  
- [ ] Final Adjustments: Optimizing Markdown and Streaming in JupyterLab  
- [ ] Multi-Shot Prompting: Enhancing LLM Reliability in AI Projects  
- [ ] Assignment: Developing Your Customized LLM-Based Tutor  
- [ ] Wrapping Up: Achievements and Next Steps

</details>

## 2: Build a Multi-Modal Chatbot

This second module teaches you how to extend LLMs beyond text by building rich, interactive chatbots. You will integrate multiple AI APIs (OpenAI, Claude, Gemini), implement real-time streaming outputs in Python, and craft adversarial conversation flows. You’ll prototype and deploy customizable UIs with Gradio-adding multi-shot prompting, context enrichment, and live function calling-to empower LLMs with external tools and code execution. Finally, you’ll combine text, images, and audio into a unified multimodal assistant, integrating DALL·E image generation, sound processing, and agentic workflows for a fully featured conversational AI.

<details>
<summary>☰ Module Content</summary>

- [ ] Multiple AI APIs: OpenAI, Claude, and Gemini for LLM Engineers  
- [ ] Streaming AI Responses: Implementing Real-Time LLM Output in Python  
- [ ] How to Create Adversarial AI Conversations Using OpenAI and Claude APIs  
- [ ] AI Tools: Exploring Transformers & Frontier LLMs for Developers  
- [ ] Building AI UIs with Gradio: Quick Prototyping for LLM Engineers  
- [ ] Gradio Tutorial: Create Interactive AI Interfaces for OpenAI GPT Models  
- [ ] Implementing Streaming Responses with GPT and Claude in Gradio UI  
- [ ] Building a Multi-Model AI Chat Interface with Gradio: GPT vs Claude  
- [ ] Building Advanced AI UIs: From OpenAI API to Chat Interfaces with Gradio  
- [ ] Building AI Chatbots: Gradio for Customer Support Assistants  
- [ ] Build a Conversational AI Chatbot with OpenAI & Gradio: Step-by-Step  
- [ ] Enhancing Chatbots with Multi-Shot Prompting and Context Enrichment  
- [ ] AI Tools: Empowering LLMs to Run Code on Your Machine  
- [ ] Using AI Tools with LLMs: Enhancing Large Language Model Capabilities  
- [ ] Building an AI Airline Assistant: Implementing Tools with OpenAI GPT-4  
- [ ] How to Equip LLMs with Custom Tools: OpenAI Function Calling Tutorial  
- [ ] AI Tools: Building Advanced LLM-Powered Assistants with APIs  
- [ ] Multimodal AI Assistants: Integrating Image and Sound Generation  
- [ ] Multimodal AI: Integrating DALL-E 3 Image Generation in JupyterLab  
- [ ] Build a Multimodal AI Agent: Integrating Audio & Image Tools  
- [ ] How to Build a Multimodal AI Assistant: Integrating Tools and Agents

</details>

## 3: Open-Source Gen AI Solutions

This module dives into open-source generative AI by guiding you through the Hugging Face ecosystem and Google Colab workflows. You’ll learn to browse and leverage community models, datasets, and Spaces; configure Colab notebooks with secure API key management; and run inference using Transformers pipelines for tasks like text generation, summarization, and joke creation. You’ll explore tokenizer architectures-LLAMA, Phi-2, Qwen, Starcoder-and compare their performance to prepare for advanced text processing. Finally, you’ll combine frontier and open-source models to build custom applications, such as an AI-powered meeting-minutes generator and a synthetic test-data generator for business use.

<details>
<summary>☰ Module Content</summary>

- [ ] Hugging Face Tutorial: Exploring Open-Source AI Models and Datasets  
- [ ] Exploring HuggingFace Hub: Models, Datasets & Spaces for AI Developers  
- [ ] Intro to Google Colab: Cloud Jupyter Notebooks for Machine Learning  
- [ ] Hugging Face Integration with Google Colab: Secrets and API Keys Setup  
- [ ] Google Colab: Run Open-Source AI Models with Hugging Face  
- [ ] Hugging Face Transformers: Using Pipelines for AI Tasks in Python  
- [ ] Hugging Face Pipelines: Simplifying AI Tasks with Transformers Library  
- [ ] HuggingFace Pipelines: Efficient AI Inference for ML Tasks  
- [ ] Exploring Tokenizers in Open-Source AI: Llama, Phi-2, Qwen, & Starcoder  
- [ ] Tokenization Techniques in AI: Using AutoTokenizer with LLAMA 3.1 Model  
- [ ] Comparing Tokenizers: Llama, PHI-3, and OWEN2 for Open-Source AI Models  
- [ ] Hugging Face Tokenizers: Preparing for Advanced AI Text Generation  
- [ ] Hugging Face Model Class: Running Inference on Open-Source AI Models  
- [ ] Hugging Face Transformers: Loading & Quantizing LLMs with Bits & Bytes  
- [ ] Hugging Face Transformers: Generating Jokes with Open-Source AI Models  
- [ ] Hugging Face Transformers: Models, Pipelines, and Tokenizers  
- [ ] Combining Frontier & Open-Source Models for Audio-to-Text Summarization  
- [ ] Using Hugging Face & OpenAI for AI-Powered Meeting Minutes Generation  
- [ ] Build a Synthetic Test Data Generator: Open-Source AI Model for Business

</details>

## 4: Model Evaluation for Code & Business Tasks  

This fourth module teaches you how to rigorously evaluate and compare language models for software development and enterprise use cases. You will learn criteria for selecting the right LLM-open or closed source-by examining scaling laws, benchmark limitations, and specialized leaderboards. Hands-on comparisons (GPT-4 vs. Claude vs. open-source models) will cover code generation performance, error analysis, and business-centric metrics. You will also build evaluation tools-such as Gradio UIs and Hugging Face endpoints-for systematic testing, discover common pitfalls, and master techniques to optimize model choice and integration for real-world code and business tasks.

<details>
<summary>☰ Module Content</summary>

- [ ] How to Choose the Right LLM: Comparing Open and Closed Source Models  
- [ ] Chinchilla Scaling Law: Optimizing LLM Parameters and Training Data Size  
- [ ] Limitations of LLM Benchmarks: Overfitting and Training Data Leakage  
- [ ] Evaluating Large Language Models: 6 Next-Level Benchmarks Unveiled  
- [ ] HuggingFace OpenLLM Leaderboard: Comparing Open-Source Language Models  
- [ ] Master LLM Leaderboards: Comparing Open Source and Closed Source Models  
- [ ] Comparing LLMs: Top 6 Leaderboards for Evaluating Language Models  
- [ ] Specialized LLM Leaderboards: Finding the Best Model for Your Use Case  
- [ ] LLAMA vs GPT-4: Benchmarking Large Language Models for Code Generation  
- [ ] Human-Rated Language Models: Understanding the LM Sys Chatbot Arena  
- [ ] Commercial Applications of Large Language Models: From Law to Education  
- [ ] Comparing Frontier and Open-Source LLMs for Code Conversion Projects  
- [ ] Leveraging Frontier Models for High-Performance Code Generation in C++  
- [ ] Comparing Top LLMs for Code Generation: GPT-4 vs Claude 3.5 Sonnet  
- [ ] Optimizing Python Code with Large Language Models: GPT-4 vs Claude 3.5  
- [ ] Code Generation Pitfalls: When Large Language Models Produce Errors  
- [ ] Blazing Fast Code Generation: How Claude Outperforms Python by 13,000x  
- [ ] Building a Gradio UI for Code Generation with Large Language Models  
- [ ] Optimizing C++ Code Generation: Comparing GPT and Claude Performance  
- [ ] Comparing GPT-4 and Claude for Code Generation: Performance Benchmarks  
- [ ] Open Source LLMs for Code Generation: Hugging Face Endpoints Explored  
- [ ] How to Use HuggingFace Inference Endpoints for Code Generation Models  
- [ ] Integrating Open-Source Models with Frontier LLMs for Code Generation  
- [ ] Comparing Code Generation: GPT-4, Claude, and CodeQuen LLMs  
- [ ] Code Generation with LLMs: Techniques and Model Selection  
- [ ] Evaluating LLM Performance: Model-Centric vs Business-Centric Metrics  
- [ ] LLM Code Generation: Advanced Challenges for Python Developers

</details>

## 5: RAG with LangChain

This module shows how to augment LLMs with external knowledge through retrieval-augmented generation. You will learn the fundamentals of RAG-why and how to leverage external data to improve response accuracy-and build a DIY RAG system from scratch. You’ll explore vector embeddings, set up OpenAI and Chroma stores, and use LangChain’s text splitters and pipeline abstractions to assemble efficient retrieval workflows. Hands-on tutorials will cover embedding visualization with t-SNE, FAISS vs. Chroma comparisons, pipeline debugging, and troubleshooting common issues. By the end, you will have built your own AI knowledge-worker capable of combining LLM reasoning with up-to-date information.

<details>
<summary>☰ Module Content</summary>

- [ ] RAG Fundamentals: Leveraging External Data to Improve LLM Responses  
- [ ] Building a DIY RAG System: Implementing Retrieval-Augmented Generation  
- [ ] Understanding Vector Embeddings: The Key to RAG and LLM Retrieval  
- [ ] Unveiling LangChain: Simplify RAG Implementation for LLM Applications  
- [ ] LangChain Text Splitter Tutorial: Optimizing Chunks for RAG Systems  
- [ ] Preparing for Vector Databases: OpenAI Embeddings and Chroma in RAG  
- [ ] Vector Embeddings: OpenAI and Chroma for LLM Engineering  
- [ ] Visualizing Embeddings: Exploring Multi-Dimensional Space with t-SNE  
- [ ] Building RAG Pipelines: From Vectors to Embeddings with LangChain  
- [ ] Implementing RAG Pipeline: LLM, Retriever, and Memory in LangChain  
- [ ] Retrieval-Augmented Generation: Hands-On LLM Integration  
- [ ] Master RAG Pipeline: Building Efficient RAG Systems  
- [ ] Optimizing RAG Systems: Troubleshooting and Fixing Common Problems  
- [ ] Switching Vector Stores: FAISS vs Chroma in LangChain RAG Pipelines  
- [ ] Demystifying LangChain: Behind-the-Scenes of RAG Pipeline Construction  
- [ ] Debugging RAG: Optimizing Context Retrieval in LangChain  
- [ ] Build Your Personal AI Knowledge Worker: RAG for Productivity Boost

</details>

## 6: Fine-Tuning with LoRA/QLoRA

This module covers end-to-end fine-tuning of large language models using parameter-efficient methods. You will learn how to source and curate balanced datasets, apply scrubbing techniques, and engineer features for LLM training. You’ll compare model- and business-centric evaluation metrics, build baseline ML models, and analyze price-description correlations. Hands-on tutorials will guide you through preparing JSONL files, launching OpenAI fine-tuning jobs, and tracking training progress with Weights & Biases. Finally, you’ll explore challenges in loss monitoring, hyperparameter optimization for LoRA/QLoRA, and best practices for productionizing fine-tuned models at scale.

<details>
<summary>☰ Module Content</summary>

- [ ] Fine-Tuning Large Language Models: From Inference to Training  
- [ ] Finding and Crafting Datasets for LLM Fine-Tuning: Sources & Techniques  
- [ ] Data Curation Techniques for Fine-Tuning LLMs on Product Descriptions  
- [ ] Optimizing Training Data: Scrubbing Techniques for LLM Fine-Tuning  
- [ ] Evaluating LLM Performance: Model-Centric vs Business-Centric Metrics  
- [ ] LLM Deployment Pipeline: From Business Problem to Production Solution  
- [ ] Prompting, RAG, and Fine-Tuning: When to Use Each Approach  
- [ ] Productionizing LLMs: Best Practices for Deploying AI Models at Scale  
- [ ] Optimizing Large Datasets for Model Training: Data Curation Strategies  
- [ ] How to Create a Balanced Dataset for LLM Training: Curation Techniques  
- [ ] Finalizing Dataset Curation: Analyzing Price-Description Correlations  
- [ ] How to Create and Upload a High-Quality Dataset on HuggingFace  
- [ ] Feature Engineering and Bag of Words: Building ML Baselines for NLP  
- [ ] Baseline Models in ML: Implementing Simple Prediction Functions  
- [ ] Feature Engineering Techniques for Amazon Product Price Prediction  
- [ ] Optimizing LLM Performance: Advanced Feature Engineering Strategies  
- [ ] Linear Regression for LLM Fine-Tuning: Baseline Model Comparison  
- [ ] Bag of Words NLP: Implementing Count Vectorizer for Text Analysis in ML  
- [ ] Support Vector Regression vs Random Forest: Machine Learning Face-Off  
- [ ] Comparing Traditional ML Models: From Random to Random Forest  
- [ ] Evaluating Frontier Models: Comparing Performance to Baseline Frameworks  
- [ ] Human vs AI: Evaluating Price Prediction Performance in Frontier Models  
- [ ] GPT-4o Mini: Frontier AI Model Evaluation for Price Estimation Tasks  
- [ ] Comparing GPT-4 and Claude: Model Performance in Price Prediction Tasks  
- [ ] Frontier AI Capabilities: LLMs Outperforming Traditional ML Models  
- [ ] Fine-Tuning LLMs with OpenAI: Preparing Data, Training, and Evaluation  
- [ ] How to Prepare JSONL Files for Fine-Tuning Large Language Models (LLMs)  
- [ ] Step-by-Step Guide: Launching GPT Fine-Tuning Jobs with OpenAI API  
- [ ] Fine-Tuning LLMs: Track Training Loss & Progress with Weights & Biases  
- [ ] Evaluating Fine-Tuned LLMs Metrics: Analyzing Training & Validation Loss  
- [ ] LLM Fine-Tuning Challenges: When Model Performance Doesn't Improve  
- [ ] Fine-Tuning Frontier LLMs: Challenges & Best Practices for Optimization

</details>

## 7: Fine-Tuned Price Prediction Model

This module delivers a deep dive into building a high-performance price-prediction model through parameter-efficient fine-tuning techniques. You will master LoRA and QLoRA adaptors, learn advanced quantization methods (8-bit, NF4), and analyze their impact on model size and accuracy. Guided tutorials cover selecting and tokenizing the optimal base model, configuring SFTTrainer for 4-bit fine-tuning, and tuning epochs, batch sizes, learning rates, and optimizers. You will launch QLoRA training jobs, monitor loss and performance with Weights & Biases, apply cost-saving strategies like smaller datasets, and visualize training metrics. Finally, you will evaluate your proprietary fine-tuned LLM against GPT-4 and business benchmarks, refining hyperparameters to achieve best-in-class price-prediction results.

<details>
<summary>☰ Module Content</summary>

- [ ] Parameter-Efficient Fine-Tuning: LoRa, QLoRA & Hyperparameters  
- [ ] Introduction to LoRA Adaptors: Low-Rank Adaptation Explained  
- [ ] QLoRA: Quantization for Efficient Fine-Tuning of Large Language Models  
- [ ] Optimizing LLMs: R, Alpha, and Target Modules in QLoRA Fine-Tuning  
- [ ] Parameter-Efficient Fine-Tuning: PEFT for LLMs with Hugging Face  
- [ ] How to Quantize LLMs: Reducing Model Size with 8-bit Precision  
- [ ] Double Quantization & NF4: Advanced Techniques for 4-Bit LLM Optimization  
- [ ] Exploring PEFT Models: The Role of LoRA Adapters in LLM Fine-Tuning  
- [ ] Model Size Summary: Comparing Quantized and Fine-Tuned Models  
- [ ] How to Choose the Best Base Model for Fine-Tuning Large Language Models  
- [ ] Selecting the Best Base Model: Analyzing HuggingFace's LLM Leaderboard  
- [ ] Exploring Tokenizers: Comparing LLAMA, OWEN, and Other LLM Models  
- [ ] Optimizing LLM Performance: Loading and Tokenizing Llama 3.1 Base Model  
- [ ] Quantization Impact on LLMs: Analyzing Performance Metrics and Errors  
- [ ] Comparing LLMs: GPT-4 vs LLAMA 3.1 in Parameter-Efficient Tuning  
- [ ] QLoRA Hyperparameters: Fine-Tuning for Large Language Models  
- [ ] Understanding Epochs and Batch Sizes in Model Training  
- [ ] Learning Rate, Gradient Accumulation, and Optimizers Explained  
- [ ] Setting Up the Training Process for Fine-Tuning  
- [ ] Configuring SFTTrainer for 4-Bit Quantized LoRA Fine-Tuning of LLMs  
- [ ] Fine-Tuning LLMs: Launching the Training Process with QLoRA  
- [ ] Monitoring and Managing Training with Weights & Biases  
- [ ] Keeping Training Costs Low: Efficient Fine-Tuning Strategies  
- [ ] Efficient Fine-Tuning: Using Smaller Datasets for QLoRA Training  
- [ ] Visualizing LLM Fine-Tuning Progress with Weights and Biases Charts  
- [ ] Advanced Weights & Biases Tools and Model Saving on Hugging Face  
- [ ] End-to-End LLM Fine-Tuning: From Problem Definition to Trained Model  
- [ ] The Four Steps in LLM Training: From Forward Pass to Optimization  
- [ ] QLoRA Training Process: Forward Pass, Backward Pass and Loss Calculation  
- [ ] Understanding Softmax and Cross-Entropy Loss in Model Training  
- [ ] Monitoring Fine-Tuning: Weights & Biases for LLM Training Analysis  
- [ ] Revisiting the Podium: Comparing Model Performance Metrics  
- [ ] Evaluation of our Proprietary, Fine-Tuned LLM against Business Metrics  
- [ ] Visualization of Results: Did We Beat GPT-4?  
- [ ] Hyperparameter Tuning for LLMs: Improving Model Accuracy with PEFT

<details>

## 8: Autonomous Multi-Agent Systems

This final module advances you from fine-tuning to architecting fully autonomous AI systems. You will design and deploy multi-agent workflows that coordinate LLMs for tasks such as automated deal finding, serverless pricing APIs, and large-scale RAG solutions. Hands-on labs cover Modal for cloud deployment, efficient LLAMA inference, and building massive Chroma datastores with 3D embedding visualizations. You’ll integrate machine-learning regressors and ensemble models with RAG pipelines, master structured outputs via Pydantic, and implement autonomous agents with planning, memory, and real-time notifications. Finally, you’ll build and scale Gradio UIs for monitoring agent performance and complete a retrospective to solidify your expertise in autonomous multi-agent AI engineering.

<details>
<summary>☰ Module Content</summary>

- [ ] From Fine-Tuning to Multi-Agent Systems: Next-Level LLM Engineering  
- [ ] Building a Multi-Agent AI Architecture for Automated Deal Finding Systems  
- [ ] Unveiling Modal: Deploying Serverless Models to the Cloud  
- [ ] LLAMA on the Cloud: Running Large Models Efficiently  
- [ ] Building a Serverless AI Pricing API: Step-by-Step Guide with Modal  
- [ ] Multiple Production Models Ahead: Preparing for Advanced RAG Solutions  
- [ ] Implementing Agentic Workflows: Frontier Models and Vector Stores in RAG  
- [ ] Building a Massive Chroma Vector Datastore for Advanced RAG Pipelines  
- [ ] Visualizing Vector Spaces: Advanced RAG Techniques for Data Exploration  
- [ ] 3D Visualization Techniques for RAG: Exploring Vector Embeddings  
- [ ] Finding Similar Products: Building a RAG Pipeline without LangChain  
- [ ] RAG Pipeline Implementation: Enhancing LLMs with Retrieval Techniques  
- [ ] Random Forest Regression: Using Transformers & ML for Price Prediction  
- [ ] Building an Ensemble Model: Combining LLM, RAG, and Random Forest  
- [ ] Wrap-Up: Finalizing Multi-Agent Systems and RAG Integration  
- [ ] Enhancing AI Agents with Structured Outputs: Pydantic & BaseModel Guide  
- [ ] Scraping RSS Feeds: Building an AI-Powered Deal Selection System  
- [ ] Structured Outputs in AI: Implementing GPT-4 for Detailed Deal Selection  
- [ ] Optimizing AI Workflows: Refining Prompts for Accurate Price Recognition  
- [ ] Autonomous Agents: Designing Multi-Agent AI Workflows  
- [ ] The 5 Hallmarks of Agentic AI: Autonomy, Planning, and Memory  
- [ ] Building an Agentic AI System: Integrating Pushover for Notifications  
- [ ] Implementing Agentic AI: Creating a Planning Agent for Automated Workflows  
- [ ] Building an Agent Framework: Connecting LLMs and Python Code  
- [ ] Completing Agentic Workflows: Scaling for Business Applications  
- [ ] Autonomous AI Agents: Building Intelligent Systems Without Human Input  
- [ ] AI Agents with Gradio: Advanced UI Techniques for Autonomous Systems  
- [ ] Finalizing the Gradio UI for Our Agentic AI Solution  
- [ ] Enhancing AI Agent UI: Gradio Integration for Real-Time Log Visualization  
- [ ] Analyzing Results: Monitoring Agent Framework Performance  
- [ ] AI Project Retrospective: Journey to Becoming an LLM Engineer

</details>

## Source

Content adapted from the [LLM Engineering: Master AI and Large Language Models](https://www.udemy.com/course/llm-engineering-master-ai-and-large-language-models/) course on Udemy.
