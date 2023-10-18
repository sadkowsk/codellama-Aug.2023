# code-llama

        B. Rozière et al., “Code Llama: Open Foundation Models for Code,” Meta AI, p. 47, Aug. 2023.

Impact of Code Llama since release: media coverage, downloads, etc.

Purpose of this ReadMe: review the 24 Aug. 2023 publication

What to look for in this presentation (discussion Q&A)
* Q1: What are the key differences between the three Code Llama models? (Code Llama; Code Llama-Python; Code Llama-Instruct)
* Q2: What legal/ethical risks remain with releasing code LLMs?

## 1. Overview
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
* Code Llama exceeds Llama 2 on MBPP/HumanEval/MultiPL-E
* Infilling provides new generation mode with small cost
* Models leverage contexts up to 100k tokens
* Instruction tuning improves safety/social behavior

### 1-e. Critique
* Limited analysis of tradeoffs between generality and code specialization
* More careful tuning may improve infilling performance
* Safety evaluations are narrow and require further auditing
* Broader assessment of long context benefits needed

## 2. Demonstration

## 3. Quick Start

## 4. Related Links
### 4-a. Meta Publications
* Meta AI > ["Code Llama: Open Foundation Models for Code"](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)
* Meta AI > ["Llama 2: Open Foundation and Fine-Tuned Chat Models"](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)
* GitHub > [Llama 2](https://github.com/facebookresearch/llama)
* GitHub > [Code Llama](https://github.com/facebookresearch/codellama)
* Hugging Face > [Code Llama](https://huggingface.co/codellama)
* Hugging Face > Llama 2 Blog: https://huggingface.co/blog/llama2
* Hugging Face > Code Llama Blog: https://huggingface.co/blog/codellama
