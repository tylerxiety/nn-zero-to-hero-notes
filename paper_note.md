Mistral 7B
- grouped-query attention (GQA) for faster inference
- sliding window attention (SWA) to effectively handle sequences of arbitrary length with a reduced inference cost
- Rolling Buffer Cache
- Pre-fill and Chunking

deployment:
- vLLM
- SkyPilot


LLaMA2
- pretraining: 
1. Pretraining Data: 2 trillion tokens
2. Training Details:
    - standard transformer
    - pre-normalization using RMSNorm
    - SwiGLU activation function
    - rotary positional embeddings
    - increased context length: 4k
    - grouped-query attention (GQA)
    - Hyperparameters:
        - AdamW optimizer, β1 =0.9, β2 = 0.95, eps = 10−5.
        - cosine learning rate schedule
        - warmup of 2000 steps
        - decay final learning rate down to 10% of the peak learning rate
        - weight decay of 0.1
        - gradient clipping of 1.0
        - a global batch-size of 4M tokens
    - Tokenizer: 
        - bytepair encoding (BPE) algorithm using the implementation from SentencePiece
        - split all numbers into individual digits
        - use bytes to decompose unknown UTF-8 characters
        - vocabulary size is 32k tokens
    - Training Hardware:
        - Meta’s Research Super Cluster, with InfiniBand,
        - internal production clusters, with RoCE
        - 3.3M GPU hours
3. Fine-tuning
    - 3.1 Supervised Fine-Tuning (SFT)
        - started with publicly available instruction tuning data
        - 27,540 high quality vendor-based annotation data
        - training Details:
            - cosine learning rate schedule with an initial learning rate of 2 × 10−5
            - a weight decay of 0.1
            - batch size 64
            - sequence length 4096 tokens
            - 2 epochs
            - concatenate all the prompts and answers with a special token to separate the prompt and answer segments, utilize an autoregressive objective and zero-out the loss on tokens from the user prompt so that backpropagate only on answer tokens.
    - 3.2 Reinforcement Learning with Human Feedback (RLHF)
        - 3.2.1 Human Preference Data Collection
            - binary comparison
            - two responses to a given prompt are sampled from two different model variants, and varying the temperature hyper-parameter
            - label the degree of preference
            - open source and internal data, prevent reward hacking
        - 3.2.2 Reward Modeling
            - 2 separate reward models for Helpfulness and safety
            -  replace the classification head with a regression head, the rest model architecture and hyper-parameters are identical
            - binary ranking loss with a margin component
            - different mixing recipes for both Helpfulness and Safety reward models
            - Training Details:
                -  1 epoch, prevent over-fitting
                -  same optimizer as for the base model
                - maximum learning rate 5 × 10−6 for the 70B parameter Llama 2-Chat and 1 × 10−5 for the rest, decreased on a cosine learning rate schedule, down to 10% of the maximum learning rate
                - warm-up of 3% of the total number of steps, with a minimum of 5.
                - batch size fixed at 512 pairs, or 1024 rows per batch
        - 3.2.3 Iterative Fine-Tuning
            - Proximal Policy Optimization (PPO)
            - Rejection Sampling fine-tuning.
    3.3 System Message for Multi-Turn Consistency
        - Ghost Attention (GAtt)
4. safety
    - 4.1 in pretaining
    - 4.2 Safety Fine-Tuning
        - Safety Supervised Fine-Tuning
        - Safety RLHF
        - Context Distillation for safety
    - 4.3 Red Teaming
5. Discussion
    - interesting learnings:
        - RL from humans and LLMs
        - RLHF learns to adapt the temperature with regard to the type of prompt





Gemini 1.5
Model Architecture
    - sparse mixture-of-expert (MoE) Transformer-based
    - MoE models use a learned routing function to direct inputs to a subset of the model’s parameters for processing


Training Infrastructure and Dataset
    - TPUv4 accelerators
    - multimodal and multilingual data for pre-training, instruction-tuning and further tuning based on human preference(not mentioning RL?)


MoE paper
