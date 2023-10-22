# "Code Llama: Open Foundation Models for Code"
**Cite the article:**
```
B. Rozière et al., “Code Llama: Open Foundation Models for Code.” arXiv, Aug. 25, 2023. [Online]. Available: http://arxiv.org/abs/2308.12950
```

> [!NOTE]
> **Focus Q&A**
> 1. _ELI5: What are Code Llama, Code Llama-Python, and Code Llama-Instruct?_
> 2. _After long context fine-tuning of Llama 2, what maximum context length can Code Llama support?_

## 1. Article Overview

> _tl;dr: The article demonstrates how the generalist LLM Llama 2 was specialized for code, handling new modalities like auto-complete infilling, and improving user-friendliness and social behavior._

### Problem
Code-specialized LLMs show better performance in code generation than general-purpose LLMs. However, code specialization also limits models maximum context lengths and neglects their potential for code infilling.

### Question
Can an open-foundation LLM be specialized for code while adding new abilities like long context processing and code infilling?

### Context
Code Llama, released by Meta AI in Aug. 2023, includes a family of three distinct models that specialize in code generation. Based on the open-foundation LLM Llama 2, the Code Llama models underwent multiple additional stages of code training and long context and instruction fine-tuning.

| *Code Llama Family Breakdown*  | Code Llama  | Code Llama—Instruct  | Code Llama—Python  |
| ------------- | :---: | :---: | :---: |
| **Domain Specialization** | General-purpose code | Human instructions for code | Python-specific code |
| **Parameter Variants** | 7B, 13B, 34B | 7B, 13B, 34B | 7B, 13B, 34B |
| **Infilling Capable** | 7B✅ 13B✅ 34B❌ | 7B✅ 13B✅ 34B❌ | 7B❌ 13B❌ 34B❌ |
| **Maximum Context Length** | 100k tokens 🔥 | 100k tokens 🔥 | 100k tokens 🔥 |

### Approach
The following subsections _A-D_ loosely reflect the Aug. 2023 article’s Section 2, “Code Llama: Specializing Llama 2 for code,”[^1] explaining how the three Code Llama variants are trained for their differing sizes and specializations. Overall, the training process involves consideration of model performance, flexibility, and safety.

<img width="1336" alt="Figure 2" src="https://github.com/sadkowsk/code-llama/assets/143565317/4f84da13-4a5a-4d15-8977-64306672b889">

> _"Figure 2: The Code Llama specialization pipeline. The different stages of fine-tuning annotated with the number of tokens seen during training. Infilling-capable models are marked with the ⇄ symbol."_[^1]

**_A. Initialization_**
- All Code Llama models are initialized using pretrained weights of Llama 2.
- All models' hyperparameters, such as learning rates, are proportional to their parameter sizes.

**_B. Dataset Training_**
- All models train on a 500B token domain-specific dataset (85% open-source GitHub code; 8% natural language about code; 7% general natural language), building on Llama 2's earlier training on 80B code tokens.
- 7B, 13B Code Llama and Code Llama-Instruct models also undergo infilling code training, using a 90% token masking rate to predict missing code from surrounding context and enabling applications like auto-complete.
- 7B, 13B, 34B Code Llama-Python models separately train on an additional 100B Python code-heavy dataset.

**_D. Long Context Fine-Tuning_**
- All models' rotary positional encodings are modified to handle longer sequences of code while retaining performance on shorter ones, and then undergo long-context fine-tuning on 16,384 token sequences.
- Enables models to handle repository-level inputs of 100k tokens, far surpassing Llama 2's 4,096 token limit.

**_E. Instruction Fine-Tuning_**
- Code Llama and Code Llama-Instruct models are further fine-tuned using human instructions and Llama 2-generated code tests in sequence batches smaller than in long context fine-tuning.
- Improves both models' bias and toxicity safety, common-sense helpfulness, and overall task performance.

### Architecture
The [facebook research / code llama model card](https://github.com/facebookresearch/codellama/blob/main/MODEL_CARD.md) summarizes the Code Llama transformer architecture:

> _"Code Llama and its variants are autoregressive language models using optimized transformer architectures. Code Llama 7B and 13B additionally support infilling text generation. All models were fine-tuned with up to 16K tokens, and support up to 100K tokens at inference time."_

However, Phuong's and Hutter's "Formal Algorithms for Transformers"[^2] enables a clearer illustration of the foundational Code Llama model architecture in algorithmic pseudocode. [Claude.ai](https://claude.ai/login) was used to compare both articles "Formal Algorithms for Transformers" and "Code Llama: Open Foundation Models for Code" to generate this pseudocode. See above repository to view as a PDF.

Given Code Llama extends from Llama 2 and the original Llama LLM, further specification of the Code Llama architecture could be possible through the earlier models' publications. See "Llama 2 Article"[^3] and "Llama 1 Article"[^4] above.

> [!WARNING]
> This pseudocode was not produced by the original Code Llama authors and may include inaccuracies. Use caution when interpreting the true Code Llama architecture.

<img width="856" alt="Pseudocode" src="https://github.com/sadkowsk/code-llama/assets/143565317/f10a56af-45bf-4024-80a3-d3bce1efa99d">

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
The screenshot below demonstrates _______. [Demonstration.ipynb](Demonstration.ipynb) (above repository) offers a separate demonstration of the foundational Code Llama 7B model's ability to _______.

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

[^1]: B. Rozière et al., “Code Llama: Open Foundation Models for Code.” arXiv, Aug. 25, 2023. Accessed: Oct. 22, 2023. [Online]. Available: http://arxiv.org/abs/2308.12950
[^2]: M. Phuong and M. Hutter, “Formal Algorithms for Transformers.” arXiv, Jul. 19, 2022. Accessed: Oct. 22, 2023. [Online]. Available: http://arxiv.org/abs/2207.09238
[^3]: H. Touvron et al., “Llama 2: Open Foundation and Fine-Tuned Chat Models.” arXiv, Jul. 19, 2023. Accessed: Oct. 22, 2023. [Online]. Available: http://arxiv.org/abs/2307.09288
[^4]: H. Touvron et al., “LLaMA: Open and Efficient Foundation Language Models.” arXiv, Feb. 27, 2023. Accessed: Oct. 22, 2023. [Online]. Available: http://arxiv.org/abs/2302.13971
