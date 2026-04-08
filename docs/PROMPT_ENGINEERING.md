# Prompt Engineering Documentation

## Overview

This document details the prompt engineering strategies used in Forge AI,
including design decisions, iterations, and evaluation criteria.

## System Prompt Design

### Strategy: Role + Expertise + Rules + Format

The system prompt follows a structured pattern that constrains the LLM's behavior
while maintaining flexibility:

```
[Persona] → Who the agent is
[Expertise] → What it knows (and doesn't)
[Rules] → Behavioral constraints (chain-of-thought, simplicity-first)
[Output Format] → How to structure responses
[Negative Rules] → What NOT to do
```

### Why This Structure?

- **Persona grounding**: Reduces hallucination by anchoring to a specific role
- **Explicit expertise boundaries**: Prevents the model from overstepping (e.g., won't claim to run code)
- **Chain-of-thought rule**: Forces reasoning before recommendations, improving quality
- **Negative rules**: Most effective way to prevent common failure modes

## Few-Shot Strategy

Each tool includes few-shot examples that demonstrate expected input/output format.
This ensures consistent, structured responses regardless of the underlying LLM.

### Dataset Analysis Few-Shot

- Shows the expected profile format (shape, missing, issues)
- Demonstrates the "detection → recommendation" pattern
- Uses emoji markers for scannability

### Code Generation Few-Shot

- Always uses scikit-learn Pipeline API
- Includes inline comments explaining each step
- Shows the full workflow: load → preprocess → train → evaluate

## Temperature Settings

| Task | Temperature | Reasoning |
|------|-------------|-----------|
| Code generation | 0.0 | Deterministic, correct syntax |
| Data analysis | 0.1 | Factual, slight variation in phrasing |
| Concept explanation | 0.3 | More creative analogies |
| General conversation | 0.2 | Natural but grounded |

## Prompt Evaluation Criteria

Each prompt iteration was evaluated against:

1. **Correctness**: Does the response contain technically accurate information?
2. **Relevance**: Does it address the user's actual question?
3. **Actionability**: Can the user act on the advice immediately?
4. **Code quality**: Is generated code runnable and well-documented?
5. **Failure handling**: Does it admit uncertainty instead of hallucinating?

## Known Failure Modes & Mitigations

| Failure Mode | Mitigation |
|-------------|-----------|
| Hallucinating library functions | RAG retrieval provides grounding |
| Overly complex first recommendation | "Simplest first" rule in system prompt |
| Generic advice without data context | Tools force data-grounded responses |
| Not using tools when appropriate | Few-shot examples demonstrate tool use |
| Verbose, unfocused responses | Output format constraint in system prompt |

## Iteration Log

| Version | Change | Impact |
|---------|--------|--------|
| v1 | Basic system prompt | Low tool usage, generic responses |
| v2 | Added explicit "use your tools" rule | Tool calling improved ~40% |
| v3 | Added few-shot examples per tool | Consistent output format |
| v4 | Added negative rules ("don't do X") | Reduced overconfident recommendations |
| v5 | Added chain-of-thought instruction | Better reasoning for complex queries |
