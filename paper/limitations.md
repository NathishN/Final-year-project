# Limitations and Threats to Validity

1. **Synthetic Laboratory Domain Domination:**  
   The evaluation test set contains 85.5% images from `fruits360`. While high overall accuracy (92.11%) is achieved, accuracy on genuine market bins (`real_quality`) drops to 39.86%. Models trained on clean turntable fruit silhouettes do not generalize zero-shot to complex commercial bins without domain adaptation.

2. **Detector-Specific OOD Sensitivity and Background Confounding Hypothesis:**  
   Mahalanobis distance AUROC drops severely from 93.26% on studio backgrounds to 18.08% on market-bin images (`real_quality`), suggesting that feature covariance models may exploit clean background cues. In contrast, Energy score remains robust (92.49% on `real_quality`, 98.89% on `real_grade`). Causal attribution of the Mahalanobis failure specifically to background versus foreground features remains a hypothesis, requiring foreground-segmented validation in future work.

3. **Minority Grade Imbalance:**  
   Grade 2 (Moderate Decay) represents only 0.52% of training data ($n=208$) and 0.93% of test data ($n=119$). While Configuration B improves Grade 2 recall from 7.4% to 21.5%, overall minority grade coverage remains sparse across species.
