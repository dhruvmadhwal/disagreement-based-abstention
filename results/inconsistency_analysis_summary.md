# Inconsistency Analysis: GPT-5.1 & Gemini-2.5-Pro (Assistive Regime)

## Executive Summary

Analyzed **229 inconsistent cases** (after correcting 6 surface-form differences) across CRAG, HotPotQA, Bamboogle, and Mintaka datasets for GPT-5.1 and Gemini-2.5-Pro in the assistive setting.

### Key Finding
The inconsistencies are caused by:
1. **Inconsistent Knowledge (42%)**: Model gives different wrong answers via different reasoning paths
2. **Decomposition Helped (27%)**: Assistive got it right, open-ended got it wrong
3. **Decomposition Hurt (21%)**: Open-ended got it right, assistive got it wrong  
4. **Abstentions (4%)**: One approach abstained while the other answered
5. **Wrong Execution (3%)**: Placeholder substitution failures or malformed responses

## Detailed Breakdown

### By Failure Category

| Category | Count | Percentage |
|----------|-------|------------|
| Knowledge Inconsistency | 218 | 92.8% |
| Abstention (Assistive) | 7 | 3.0% |
| Abstention (Open-ended) | 3 | 1.3% |
| Wrong Execution | 7 | 3.0% |

### Knowledge Inconsistency Sub-Classification (by Correctness)

| Sub-Category | Count | Interpretation |
|--------------|-------|----------------|
| Both Wrong | 99 | Model has knowledge gaps - gives different wrong answers |
| Assistive Correct, Open Wrong | 61 | **Decomposition helped** - step-by-step led to correct answer |
| Assistive Wrong, Open Correct | 49 | **Decomposition hurt** - error propagation in intermediate steps |
| Unknown | 3 | Missing correctness data |

## By Dataset and Model

| Dataset/Model | Both Wrong | Asst Right | Open Right | Abstention | Wrong Exec | Total |
|---------------|------------|------------|------------|------------|------------|-------|
| bamboogle/google-gemini-2-5-pro | 3 | 4 | 1 | 0 | 2 | 10 |
| bamboogle/gpt-5-1 | 3 | 6 | 3 | 0 | 0 | 13 |
| crag/google-gemini-2-5-pro | 12 | 3 | 7 | 2 | 2 | 28 |
| crag/gpt-5-1 | 16 | 4 | 6 | 2 | 0 | 29 |
| hotpotqa/google-gemini-2-5-pro | 36 | 15 | 13 | 1 | 1 | 68 |
| hotpotqa/gpt-5-1 | 17 | 14 | 11 | 3 | 0 | 45 |
| mintaka/google-gemini-2-5-pro | 6 | 9 | 3 | 1 | 2 | 22 |
| mintaka/gpt-5-1 | 6 | 6 | 5 | 1 | 0 | 20 |

## Analysis of Why HotPotQA Has Lower Consistency

HotPotQA shows the lowest consistency rates:
- **GPT-5.1**: 78.3% (45 inconsistent)
- **Gemini-2.5-Pro**: 75.3% (68 inconsistent)

### Root Causes in HotPotQA:

1. **High "Both Wrong" Rate**: 53 cases where both approaches failed with different wrong answers
   - These questions often require precise factual recall (dates, numbers, names)
   - Example: "What year was the thesis supervisor of Jocelyn Bell Burnell awarded the Eddington Medal?"
     - Assistive: 1977, Open-ended: 1969 (both wrong, correct is different)

2. **Decomposition Errors**: 24 cases where decomposition hurt performance
   - Wrong intermediate facts propagate: "Who opened Royal Spa Centre?" → Wrong person → Wrong party
   - Comparison errors: Got both mall sizes but compared incorrectly

3. **Multi-hop complexity**: Questions require multiple correct facts to chain correctly

## Examples by Category

### Decomposition Helped (Assistive Correct, Open-ended Wrong)

**crag/gpt-5-1**: at what age was lorne michaels when he produced his first film?
- Assistive (✓): 35
- Open-ended (✗): 33
- Intermediate: {'answer_1': 'Gilda Live', 'answer_2': '35'}

**crag/gpt-5-1**: what is the top artist in the most popular music genre in 2015?
- Assistive (✓): Taylor Swift
- Open-ended (✗): Adele
- Intermediate: {'answer_1': 'Pop', 'answer_2': 'Taylor Swift'}

**crag/gpt-5-1**: who was the first actress to play the role of wonder woman in a live-action movie?
- Assistive (✓): Cathy Lee Crosby
- Open-ended (✗): Ellie Wood Walker
- Intermediate: {'answer_1': 'Wonder Woman (1974 television film)', 'answer_2': 'Cathy Lee Crosby'}

### Decomposition Hurt (Assistive Wrong, Open-ended Correct)

**crag/gpt-5-1**: how many months did it take to release the first toy story movie?
- Assistive (✗): 58
- Open-ended (✓): 48
- Intermediate: {'answer_1': '1995-11-22', 'answer_2': '1991', 'answer_3': '58'}
- **Failure mode**: Wrong intermediate answer propagated

**crag/gpt-5-1**: what's the most recent album from the puerto rican artist that's been in wwe?
- Assistive (✗): Un Verano Sin Ti
- Open-ended (✓): Nadie Sabe Lo Que Va a Pasar Mañana
- Intermediate: {'answer_1': 'Bad Bunny', 'answer_2': 'Un Verano Sin Ti'}
- **Failure mode**: Wrong intermediate answer propagated

**crag/gpt-5-1**: who is the american singer-songwriter who has won 11 grammy awards and is known for her unique voice and poignant lyrics, including her hit songs "both sides now" and "big yellow taxi"?
- Assistive (✗): There is no such single person; Taylor Swift and Joni Mitchell are two different singers
- Open-ended (✓): Joni Mitchell
- Intermediate: {'answer_1': 'Taylor Swift', 'answer_2': 'Joni Mitchell', 'answer_3': 'There is no such single person; Taylor Swift and Joni Mitchell are two different singers'}
- **Failure mode**: Wrong intermediate answer propagated

## Conclusions

1. **Most inconsistencies are NOT due to bad DSLs** - they are genuine model behavior:
   - 42% are knowledge gaps (both wrong with different answers)
   - 27% show decomposition helping
   - 21% show decomposition hurting (error propagation)

2. **HotPotQA is harder** because:
   - Questions require precise factual knowledge
   - Multi-hop reasoning amplifies small errors
   - Temporal/numerical facts are less stable in model knowledge

3. **Decomposition is a trade-off**:
   - Helps in 61 cases, hurts in 49 cases (net +12 cases improved)
   - The decomposition makes reasoning more systematic but also exposes intermediate errors

4. **Only 3% are execution failures** (placeholder errors, malformed responses)
