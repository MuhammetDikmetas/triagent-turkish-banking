# Evaluation results

Test set: **160 messages** (8 categories × 20), held out from training.  
Encoder: `tfidf` · routing threshold τ = **0.59** (chosen by 5-fold CV on train)

## Classification & priority

| System | Accuracy | Macro-F1 | FRAUD recall | Priority acc. | P1 recall | LLM calls | Latency/msg |
|---|---|---|---|---|---|---|---|
| classifier (tfidf) | 86.9% | 0.868 | 90.0% | 80.6% | 88.2% | 0% | 0.1 ms |
| classifier (tfidf) + rules | 87.5% | 0.875 | 95.0% | 91.9% | 94.1% | 0% | 0.1 ms |
| llm only | 95.0% | 0.950 | 100.0% | 91.2% | 100.0% | 100% | 2759.0 ms |
| llm + rules | 95.0% | 0.950 | 100.0% | 90.0% | 100.0% | 100% | 2759.0 ms |
| hybrid + rules (τ=0.59)  ← Triagent | 94.4% | 0.944 | 95.0% | 91.2% | 94.1% | 39% | 1086.5 ms |

## PII masking

Synthetic benchmark: 400 sentences mixing checksum-valid identifiers with look-alike decoys (order numbers, 16-digit references, customer numbers, amounts, dates).

| Type | Precision | Recall | Support |
|---|---|---|---|
| TCKN | 100.0% | 100.0% | 129 |
| IBAN | 100.0% | 100.0% | 105 |
| CARD | 96.2% | 100.0% | 128 |
| PHONE | 100.0% | 100.0% | 123 |
| EMAIL | 100.0% | 100.0% | 111 |

On the test set: 6/6 PII spans masked, 0 false positives (test set includes decoy reference numbers).

![threshold](threshold_curve.png)
![confusion](confusion_matrix.png)
