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

<img width="1336" alt="Figure 2" src="https://github.com/sadkowsk/codellama-Aug.2023/assets/143565317/0624631e-6e28-404d-95f7-7cc3e9a24538">

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

Phuong's and Hutter's "Formal Algorithms for Transformers"[^2] offers notation to further illustrate the foundational Code Llama model architecture in algorithmic pseudocode. [Claude.ai](https://claude.ai/login) was used to compare both articles "Formal Algorithms for Transformers" and "Code Llama: Open Foundation Models for Code" to generate this pseudocode. See above repository to view as a PDF.

Given Code Llama extends from Llama 2 and the original Llama 1 LLM, it is worth noting that additional specification of the Code Llama architecture could be possible through the earlier models' publications. See "Llama 2 Article"[^3] and "Llama 1 Article"[^4] above.

> [!WARNING]
> This pseudocode was not produced by the original Code Llama authors and may include inaccuracies. Use caution when interpreting the true Code Llama architecture.

<img width="856" alt="Pseudocode" src="https://github.com/sadkowsk/codellama-Aug.2023/assets/143565317/8a007e42-e24f-4f3e-b72e-58836c4db7a9">

### Results

Overall, each of the three Code Llama models exceeds the coding performance of Llama 2; Code Llama 7B even outperforms larger general models like CodeGen-Multi and StarCoder in multilingual evaluations, and the Code Llama-Python variants set a new SOTA on HumanEval and MBPP benchmarks. Infilling training for Code Llama and Code Llama-Instruct also successfully enables middle-out generation without significantly compromising autoregressive performance. For Code Llama-Instruct, fine-tuning with instructions enhances safety and helpfulness metrics, although with a slight decrease in raw coding scores.

Despite all these achievements, it is understandable that while the largest 34B Code Llama model variants offer the greater capacity and performance, they are also more computationally expensive to train and deploy with higher latency.

The following _Table 2_ and _Figure 3_ from the Aug. 2023 article illustrate Code Llama's benchmarking and multilingual achievements:

<img width="736" alt="Table 2" src="https://github.com/sadkowsk/codellama-Aug.2023/assets/143565317/a151892c-1aae-44d9-9050-1e9ef79be02c">

> _"Table 2: Code Llama pass@ scores on HumanEval and MBPP. The pass@1 scores of our models are computed with greedy decoding. The pass@10 and pass@100 scores are computed with nucleus sampling with p=0.95 and temperature 0.8 following our findings from Figure 6. Models are evaluated in zero-shot on Human Eval and 3-shot on MBPP. The instruct models are trained to be safe and aligned from the base Code Llama models. Results for other models as provided by Li et al. (2023) (code-cushman-001, StarCoder), OpenAI (2023) (GPT-3.5, GPT-4), and Chowdhery et al. (2022); Anil et al. (2023) (PaLM)."_[^1]

<img width="1358" alt="Figure 3" src="https://github.com/sadkowsk/codellama-Aug.2023/assets/143565317/8ff99b43-ca8d-421d-a134-3ff75355daa5">

> _"Figure 3: Correlations between Languages. Correlation scores between the Python, C++, Java, PHP, C#, TypeScript (TS), and Bash, reported for different model sizes. The code for this figure was generated by Code Llama - Instruct, the prompt and code can be seen in Figure 21."_[^1]

### Critique
* Limited analysis of the tradeoffs between training general purpose models versus specializing models for code warrants further investigation.
* More careful tuning of the training hyperparameters may lead to improved performance on infilling tasks.
* While preliminary safety evaluations were conducted, they cover only a narrow set of potential harms.
* A broader assessment using diverse real-world tasks is needed to conclusively demonstrate the benefits of long context handling.

## 2. Code Llama Demonstration
Figure 16 below shows several prompts and responses during a user interaction with Code Llama-Instruct. [Demonstration.ipynb](Demonstration.ipynb) (above repository) also offers a limited demonstration of the foundational Code Llama 7B model running locally.

<img width="877" alt="Figure 16" src="https://github.com/sadkowsk/codellama-Aug.2023/assets/143565317/556d6575-fe1a-4e1a-9b55-13a55a0b2c7d">

> _"Figure 16: Example of standard python bugs found and explained by Code Llama - Instruct."_

## 3. Code Llama Quick Start
For Mac users wishing to run Code Llama locally with VS Code, download Ollama from [jmorganca / ollama](https://github.com/jmorganca/ollama) and follow the contained how-to. Happy coding!

## 4. External Links
* GitHub > [facebookresearch / codellama](https://github.com/facebookresearch/codellama)
* GitHub > [facebook research / llama2](https://github.com/facebookresearch/llama)
* Hugging Face > [Meta / Code Llama](https://huggingface.co/codellama)
* Hugging Face > [Meta / Llama 2](https://huggingface.co/meta-llama)
* Meta AI > ["Code Llama: Open Foundation Models for Code"](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)
* Meta AI > ["Llama 2: Open Foundation and Fine-Tuned Chat Models"](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)

> [!NOTE]
> **Focus Q&A**
> 1. _ELI5: What are Code Llama, Code Llama-Python, and Code Llama-Instruct?_
> 2. _After long context fine-tuning of Llama 2, what maximum context length can Code Llama support?_

[^1]: B. Rozière et al., “Code Llama: Open Foundation Models for Code.” arXiv, Aug. 25, 2023. Accessed: Oct. 22, 2023. [Online]. Available: http://arxiv.org/abs/2308.12950
[^2]: M. Phuong and M. Hutter, “Formal Algorithms for Transformers.” arXiv, Jul. 19, 2022. Accessed: Oct. 22, 2023. [Online]. Available: http://arxiv.org/abs/2207.09238
[^3]: H. Touvron et al., “Llama 2: Open Foundation and Fine-Tuned Chat Models.” arXiv, Jul. 19, 2023. Accessed: Oct. 22, 2023. [Online]. Available: http://arxiv.org/abs/2307.09288
[^4]: H. Touvron et al., “LLaMA: Open and Efficient Foundation Language Models.” arXiv, Feb. 27, 2023. Accessed: Oct. 22, 2023. [Online]. Available: http://arxiv.org/abs/2302.13971
