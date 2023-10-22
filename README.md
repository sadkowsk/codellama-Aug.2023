# "Code Llama: Open Foundation Models for Code"
**Cite the article:**
```
B. Rozière et al., “Code Llama: Open Foundation Models for Code,” Meta AI, p. 47, Aug. 2023.
```

> [!NOTE]
> **Focus Q&A**
> 1. _ELI5: What are Code Llama, Code Llama-Python, and Code Llama-Instruct?_
> 2. _After long context fine-tuning, what maximum context lengths can Llama 2 and Code Llama support?_

## 1. Article Overview

_tl;dr: The paper demonstrates methods to adapt a generalist LLM to specialized domains like code, handling new modalities like infilling, and improving social behavior._

### Problem
Code generation is a common application of large language models (LLMs), and code-specialized models like Codex and AlphaCode show better code performance than general LLMs like GPT-3. However, the capabilities of these code-specialized models are impacted by their limited context sizes and lack of infilling.

### Question
Can an open, general LLM be specialized for code while adding new abilities like long context and infilling?

### Context
Fine-tuned on the Llama 2 foundation LLM, in Aug. 2023 Meta AI released three Code Llama variant models that specialize in code:
* Code Llama: general-purpose code
* Code Llama-Python: Python code
* Code Llama-Instruct: fine-tuned for human instructions

Code Llama specializes the general foundation Llama 2 model for code tasks by pretraining on code data. The variants further improve Python specificity and safety/helpfulness.

### Approach
* Fine-tune Llama 2 foundation model on code data
* Add infilling objective for 7B and 13B models
* Long context fine-tuning to handle 100k tokens
* Release Code Llama base and Python/Instruct variants

<img width="1336" alt="Figure 2" src="https://github.com/sadkowsk/code-llama/assets/143565317/4f84da13-4a5a-4d15-8977-64306672b889">

> _"Figure 2: The Code Llama specialization pipeline. The different stages of fine-tuning annotated with the number of tokens seen during training. Infilling-capable models are marked with the ⇄ symbol."_[^1]

### Architecture
The pseudocode below follows Phuong's and Hutter's "Formal Algorithms for Transformers"[^2] to illustrate the architecture of the foundational Code Llama model. [Claude.ai](https://claude.ai/login) was used to compare both articles "Formal Algorithms for Transformers" and "Code Llama: Open Foundation Models for Code" to generate this pseudocode. See above to view this pseudocode as a PDF.

Please read with caution; this pseudocode is not by the original Code Llama authors and may include errors.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img width="856" alt="Pseudocode" src="https://github.com/sadkowsk/code-llama/assets/143565317/f10a56af-45bf-4024-80a3-d3bce1efa99d">

### Results
* Code Llama exceeds Llama 2 on MBPP/HumanEval/MultiPL-E

<img width="736" alt="Table 2" src="https://github.com/sadkowsk/code-llama/assets/143565317/076ed2a6-53b1-42e2-873c-eecef37cb26b">

> _"Table 2: Code Llama pass@ scores on HumanEval and MBPP. The pass@1 scores of our models are computed with greedy decoding. The pass@10 and pass@100 scores are computed with nucleus sampling with p=0.95 and temperature 0.8 following our findings from Figure 6. Models are evaluated in zero-shot on Human Eval and 3-shot on MBPP. The instruct models are trained to be safe and aligned from the base Code Llama models. Results for other models as provided by Li et al. (2023) (code-cushman-001, StarCoder), OpenAI (2023) (GPT-3.5, GPT-4), and Chowdhery et al. (2022); Anil et al. (2023) (PaLM)."_[^1]

* Infilling provides new generation mode with small cost
* Models leverage contexts up to 100k tokens

* Instruction tuning improves safety/social behavior

<img width="591" alt="Table 9" src="https://github.com/sadkowsk/code-llama/assets/143565317/ea05d15d-a0c9-466e-ab8d-a3eb0365d92e">

> _"Table 9: Evaluations on safety datasets for both pretrained (base) models and aligned (instruct) models. For TruthfulQA, we present the percentage of generations that are both truthful and informative (the higher, the better). For ToxiGen, we present the percentage of toxic generations (the smaller, the better). For BOLD, we present the average sentiment scores across demographic groups. A score closer to 0 indicates a neutral sentiment, while a positive (negative) score indicates a positive (negative) sentiment towards the population mentioned in the prompt."_[^1]

### Critique
* Limited analysis of tradeoffs between generality and code specialization
* More careful tuning may improve infilling performance
* Safety evaluations are narrow and require further auditing
* Broader assessment of long context benefits needed

## 2. Code Llama Demonstration
The screenshot below demonstrates _______. [Demonstration.ipynb](Demonstration.ipynb) (stored above) offers a separate demonstration of the foundational Code Llama 7B model's ability to _______.

## 3. Code Llama Quick Start
For Mac users wishing to run Code Llama locally with VS Code:
1. Download Ollama: [jmorganca / ollama](https://github.com/jmorganca/ollama)
2. Run through LangChain...?
3. Begin Jupyter Notebook file with code block:
```
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler                                  
llm = Ollama(model="codellama", 
             callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]))
```
4. In Terminal, input:
```
ollama run codellama
```
5. Happy coding!
## 4. External Links
* GitHub > [facebookresearch / codellama](https://github.com/facebookresearch/codellama)
* GitHub > [facebook research / llama2](https://github.com/facebookresearch/llama)
* Hugging Face > [Meta / Code Llama](https://huggingface.co/codellama)
* Hugging Face > [Meta / Llama 2](https://huggingface.co/meta-llama)
* Meta AI > ["Code Llama: Open Foundation Models for Code"](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)
* Meta AI > ["Llama 2: Open Foundation and Fine-Tuned Chat Models"](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)

[^1]: B. Rozière et al., “Code Llama: Open Foundation Models for Code,” Meta AI, Aug. 2023, Accessed: Oct. 17, 2023. [Online]. Available: https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/
[^2]: M. Phuong and M. Hutter, “Formal Algorithms for Transformers.” arXiv, Jul. 19, 2022. Accessed: Oct. 22, 2023. [Online]. Available: http://arxiv.org/abs/2207.09238
