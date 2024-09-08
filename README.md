# LLM Supervised finetuning with PEFT

- On MultiNLI dataset (nyu-mll/multi_nli)
- Tokenization

  Experienced with finetuning methods:
  - AdaLoRA (AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning)
  - Prefix tuning

 ## AdaLoRA Configuration

### Performance Metrics

#### Validation Matched Split
| Metric      | Untrained | Trained | Difference |
|-------------|-----------|---------|------------|
| Accuracy    | 0.3442    | 0.6354  | +0.2912    |
| Precision   | 0.2461    | 0.6348  | +0.3887    |
| Recall      | 0.3442    | 0.6354  | +0.2912    |
| F1          | 0.2434    | 0.6346  | +0.3912    |


## Prefix Tuning Configuration

#### Validation Matched Split

| Metric      | Untrained | Trained | Difference |
|-------------|-----------|---------|------------|
| Accuracy    | 0.3055    | 0.7026  | +0.3971    |
| Precision   | 0.2795    | 0.7041  | +0.4246    |
| Recall      | 0.3055    | 0.7026  | +0.3971    |
| F1          | 0.1536    | 0.7032  | +0.5495    |
