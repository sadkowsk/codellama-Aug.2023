# "Code Llama: Open Foundation Models for Code"
**Cite the article:**

    B. Rozière et al., “Code Llama: Open Foundation Models for Code,” Meta AI, p. 47, Aug. 2023.

**Focus Q&A:**
1. What distinguishes each of the three Code Llama models? (Code Llama; Code Llama-Python; Code Llama-Instruct)
2. What legal/ethical risks remain with releasing code LLMs?

## 1. Article Overview
### Problem
Code generation is a common application of large language models (LLMs), and code-specialized models like Codex and AlphaCode show better code performance than general LLMs like GPT-3. However, the capabilities of these code-specialized models are impacted by their limited context sizes and lack of infilling.

### Question
Can an open, general LLM be specialized for code while adding new abilities like long context and infilling?

### Approach
* Fine-tune Llama 2 foundation model on code data
* Add infilling objective for 7B and 13B models
* Long context fine-tuning to handle 100k tokens
* Release Code Llama base and Python/Instruct variants
* (What's the "most important part of the paper?")

"Figure 2: The Code Llama specialization pipeline. The different stages of fine-tuning annotated with the number of tokens seen during training. Infilling-capable models are marked with the ⇄ symbol." (Code Llama, p.3)
    <img width="1090" alt="Screenshot 2023-10-18 at 12 42 15" src="https://github.com/sadkowsk/code-llama/assets/143565317/78775c6e-95df-4f97-9311-53f0a0033510">

### Architecture

### Results
* Code Llama exceeds Llama 2 on MBPP/HumanEval/MultiPL-E
* Infilling provides new generation mode with small cost
* Models leverage contexts up to 100k tokens
* Instruction tuning improves safety/social behavior

### Critique
* Limited analysis of tradeoffs between generality and code specialization
* More careful tuning may improve infilling performance
* Safety evaluations are narrow and require further auditing
* Broader assessment of long context benefits needed

## 2. Code Llama Demonstration
[Demonstration.ipynb](Demonstration.ipynb)

## 3. Code Llama Quick Start
For Mac users trying to run Code Llama locally on VS Code:
1. Download Ollama: [jmorganca / ollama](https://github.com/jmorganca/ollama)
2. Run through LangChain...?
3. Begin Jupyter Notebook file with command:
```
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler                                  
llm = Ollama(model="codellama", 
             callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]))
```
## 4. External Links
* GitHub > [facebookresearch / codellama](https://github.com/facebookresearch/codellama)
* GitHub > [facebook research / llama2](https://github.com/facebookresearch/llama)
* Hugging Face > [Meta / Code Llama](https://huggingface.co/codellama)
* Hugging Face > [Meta / Llama 2](https://huggingface.co/meta-llama)
* Meta AI > ["Code Llama: Open Foundation Models for Code"](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)
* Meta AI > ["Llama 2: Open Foundation and Fine-Tuned Chat Models"](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)
