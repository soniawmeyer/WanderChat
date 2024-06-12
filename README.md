# WanderChat Project README

This GitHub is a subset of the work presented in this master's thesis project:

S. Meyer, A. Ren, S. Singh, B. Tam, and C. Ton, "A Comparison of LLM Chat Bot Implementation Methods with Travel Use Case," Project report, Department of Applied Data Science, San Jose State University, San Jose, CA, USA, May 8, 2024.

The original group that conducted this research is Sonia Meyer, Angel Ren, Shreya Singh, Bertha Tam, and Christopher Ton. Two of us, Sonia Meyer and Shreya Singh, decided to pursue publishing a smaller more focused subset of our research and project, focusing only on model comparisons and including this GitHub, however, all contributed to this work through the original research.

## Introduction

WanderChat is an advanced AI-assisted travel planning chatbot designed to provide personalized travel recommendations. Unlike generic chatbots, WanderChat leverages cutting-edge AI technology to offer tailored suggestions based on user preferences, enhancing the travel planning experience. WanderChat does this using an enhanced travel-specific datasets tailored from Reddit travel related subreddits.

## Technical Overview

### Dataset

* Custom travel related Reddit data (extracted via the Reddit API) [hosted on HuggingFace](https://huggingface.co/datasets/soniawmeyer/reddit-travel-QA-finetuning)

### Models

* Pretrained LLMs: [LLaMa2 7b](https://huggingface.co/meta-llama/Llama-2-7b-hf), [Mistral 7b](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
* Methods Applied: Quantized Low Rank Adapter (QLoRA), Retrieval Augmented Finetuning (RAFT), Reinforcement Learning from Human Feedback (RLHF)
	* [QLoRA LLaMa](https://huggingface.co/beraht/llama-2-7b_qlora_falcon_417)
	* [RAFT LLaMa](https://huggingface.co/beraht/Llama2_Falcon_RAFT_50e_10s/tree/main)
	* [QLoRA Mistral](https://huggingface.co/sherrys/mistral-2-7b_qlora_falcon_426/tree/main)
	* [RAFT Mistral](https://huggingface.co/sherrys/426_mistral_RAFT_50e_10s)
	* [best model (Mistral RAFT) + RLHF](https://huggingface.co/chriztopherton/Wanderchat_Mistral_RAFT_RLHF)
			  
### Evaluation

* Metrics: Traditional NLP, RAGAS, OpenAI GPT-4, Human Evaluation

## Findings:

* Quantitative and RAGAS metrics do not always align with human evaluation.
* OpenAI GPT-4 evaluation aligns closely with human evaluation.
* Human evaluation is crucial for accurate assessment.
* Mistral generally outperformed LLaMa.
* RAFT is the best method compared to QLoRA and RAG but requires postprocessing.
* RLHF significantly improves model performance.

### License

This project is licensed under the MIT License.

### Contact

For any inquiries, please contact us through GitHub.