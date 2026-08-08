# Implementation Guide: Uncertainty-Aware Iterative RAG

This guide is designed for an AI Coding Agent to implement the "Uncertainty-Aware Iterative RAG via Semantic Information Gain" methodology. Follow these step-by-step instructions to build the system robustly.

## 1. System Architecture & Prerequisites
You will need the following components:
- **Primary LLM (Generator)**: E.g., OpenAI `gpt-4o` or similar. Must support setting `temperature > 0` and returning `logprobs`.
- **Claim Extractor**: A secondary LLM call (or the same primary LLM) with a prompt designed to extract core propositions from long text.
- **NLI Model (Clustering)**: A lightweight model like `cross-encoder/nli-deberta-v3-base` (via HuggingFace) or an LLM prompt to check for bidirectional entailment.
- **Vector Database**: For document retrieval in Step 4.

## 2. Core Execution Loop

### Step 1: Semantic Uncertainty Profiling
**Goal**: Calculate $SE_{total}$, $SE_{aleatoric}$, and $\mathcal{I}_{semantic}$ (Epistemic).

1. **Sampling**: 
   - Call the Primary LLM $M$ times (e.g., $M=10$) with `temperature=0.7`, providing Query $Q$ and current Context $C_t$.
   - **Important Requirement**: Request `logprobs` in the API call.
   - You will receive $M$ sequences: $\{s_1, s_2, ..., s_M\}$ along with their token-level logprobs.
2. **Claim Extraction (Crucial for Long Texts)**:
   - For each sequence $s_i$, use a prompt to extract "Core Claims" (e.g., extracting factual statements, dropping transition/fluff words).
   - *Example Text*: "DOMA là mô hình AI dùng để tự động hóa giao diện, được huấn luyện trên 15,000 mẫu."
   - *Extracted Claims*: `["DOMA là mô hình AI", "DOMA tự động hóa giao diện", "DOMA train trên 15,000 mẫu"]`
3. **Semantic Clustering (NLI)**:
   - Use the NLI model to compare the Core Claims of every pair of sequences.
   - If sequence $A$'s claims entail sequence $B$'s claims, and vice versa, group them into the same **Concept** ($c$).
   - Result: A set of distinct Concepts $\{c_1, c_2, ...\}$ and the counts of sequences belonging to each.
4. **Calculate Total Uncertainty ($SE_{total}$)**:
   - Calculate probability via frequency: $P(c_k) = \frac{\text{Count}(s \in c_k)}{M}$.
   - Formula: $SE_{total} = - \sum (P(c_k) \times \log_2 P(c_k))$.
5. **Calculate Aleatoric Uncertainty ($SE_{aleatoric}$)**:
   - For each sequence $s_i$, identify the **Key Tokens** within its Core Claims (ignore stop words; focus on nouns, verbs, numbers).
   - Calculate the average logprob of these Key Tokens for $s_i$. Let this be $AvgLogprob(s_i)$.
   - The token entropy for $s_i$ is approximated as $\mathcal{H}(s_i) = - AvgLogprob(s_i)$.
   - Formula: $SE_{aleatoric} = \frac{1}{M} \sum_{i=1}^M \mathcal{H}(s_i)$.
6. **Calculate Epistemic Uncertainty ($\mathcal{I}_{semantic}$)**:
   - Formula: $\mathcal{I}_{semantic} = SE_{total} - SE_{aleatoric}$.

### Step 2: Routing Logic
Define hyperparameters: `TAU_STOP`, `TAU_NOISE`, `TAU_MISSING`.
- **Condition A**: If $SE_{total} <$ `TAU_STOP` $\rightarrow$ System is confident. Return the concept with the highest probability. **EXIT LOOP**.
- **Condition B**: If $SE_{total} \ge$ `TAU_STOP`:
  - Check $SE_{aleatoric}$. If $SE_{aleatoric} >$ `TAU_NOISE` $\rightarrow$ Execute **Context Pruning** (Step 3).
  - Else if $\mathcal{I}_{semantic} >$ `TAU_MISSING` $\rightarrow$ Execute **Active Retrieval** (Step 4).
  - *Conflict Resolution*: Always prioritize Context Pruning over Active Retrieval if both thresholds are exceeded.

### Step 3: Semantic Context Pruning (Noise Handling)
**Goal**: Remove contradictory or noisy chunks from the context.
1. Iterate over each chunk $c_i$ in the current context $C_t$.
2. Temporarily remove $c_i$ to create a trial context: $C_{trial} = C_t \setminus \{c_i\}$.
3. Re-run **Step 1** (Sampling & $SE_{total}$ calculation) using $C_{trial}$. Let this result be $SE_{trial}$.
4. Calculate Marginal Information Gain: $\Delta SE = SE_{trial} - SE_{total\_original}$.
5. If $\Delta SE \le 0$: The chunk $c_i$ is noisy or unhelpful. Permanently drop it.
6. If $\Delta SE > 0$: Keep chunk $c_i$.
7. The remaining chunks form $C_{clean}$. 
8. Route to **Step 4** (if Epistemic is high) or loop back to Step 1.

### Step 4: Look-ahead Active Retrieval (Knowledge Handling)
**Goal**: Fetch missing knowledge based on the most likely hypothesis.
1. Take the Concept $c$ with the highest probability $P(c)$ from Step 1.
2. Use this Concept's core claims to form a hypothetical document/query $d'$.
3. Evaluate Expected Information Gain (EIG): (If strictly following the paper, check if $SE_{total}$ drops when adding $d'$ to $C_{clean}$).
4. Use $d'$ (the core claims) as the search query against the Vector Database.
5. Retrieve the top-$k$ new document chunks ($d_{new}$).
6. Update context: $C_{next} = C_{clean} \cup \{d_{new}\}$.
7. Loop back to **Step 1** with the new context $C_{next}$.

## 3. Recommended Code Structure / Pseudo-code

```python
class Concept:
    def __init__(self):
        self.sequences = [] # list of generated strings
        self.probability = 0.0

def run_iterative_rag(query: str, initial_context: list):
    context = initial_context
    
    while True:
        # 1. Sample and calculate uncertainties
        samples = generate_samples_with_logprobs(query, context, M=10, temperature=0.7)
        claims = [extract_claims(s) for s in samples]
        concepts = cluster_with_nli(claims)
        
        se_total = calculate_se_total(concepts)
        se_aleatoric = calculate_se_aleatoric(samples, claims)
        se_epistemic = se_total - se_aleatoric
        
        # 2. Routing
        if se_total < TAU_STOP:
            return get_best_answer(concepts)
            
        if se_aleatoric > TAU_NOISE:
            context = prune_context(query, context)
            
        elif se_epistemic > TAU_MISSING:
            best_hypothesis = get_highest_prob_concept(concepts)
            new_docs = retrieve_from_vectordb(best_hypothesis)
            context.extend(new_docs)
```
