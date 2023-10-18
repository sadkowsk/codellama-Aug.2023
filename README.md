# code-llama
Impact of Code Llama since release: media coverage, downloads, etc.

Purpose of this ReadMe: review of the 24 Aug. 2023 publication

Full article citation: 

## 1. Overview
tl;dr: 

What to look for in this presentation (discussion Q&A)
1. Q1
2. Q2

### 1-a. Context
Code generation is a common application of large language models (LLMs), and code-specialized models like Codex and AlphaCode show better code performance than general LLMs like GPT-3. However, the capabilities of these code-specialized models are impacted by their limited context sizes and lack of infilling.

### 1-b. Question
Can an open, general LLM be specialized for code while adding new abilities like long context and infilling?

### 1-c. Approach
* Fine-tune Llama 2 foundation model on code data
* Add infilling objective for 7B and 13B models
* Long context fine-tuning to handle 100k tokens
* Release Code Llama base and Python/Instruct variants
* (What's the "most important part of the paper?")

"Figure 2: The Code Llama specialization pipeline. The different stages of fine-tuning annotated with the number of tokens seen during training. Infilling-capable models are marked with the ⇄ symbol." (Code Llama, p.3)
    <img width="1090" alt="Screenshot 2023-10-18 at 12 42 15" src="https://github.com/sadkowsk/code-llama/assets/143565317/78775c6e-95df-4f97-9311-53f0a0033510">

### 1-d. Results
* 

### 1-e. Critique
* What was overlooked by the authors? What could have been developed further? Were there any errors? Have others disputed the findings?

## 2. Demonstration

## 3. Quick Start

## 4. Related Links
### 4-a. Meta Publications
* Llama 2 (18 Jul. 2023): https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/
* Code Llama (24 Aug. 2023): https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/
### 4-b. GitHub Repositories
* Llama: https://github.com/facebookresearch/llama
* Code Llama: https://github.com/facebookresearch/codellama
### 4-c. Hugging Face Blogs
* Llama 2: https://huggingface.co/blog/llama2
* Code Llama: https://huggingface.co/blog/codellama
### 4-e. Hugging Face Models
* Code Llama: https://huggingface.co/codellama
