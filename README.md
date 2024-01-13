# CS4248-Sarcasm-Detection

**Authors:** Sneha Kumar, Matthias Koh Yong An, Wilson Widhyadana, Low Kang Ngee, Kaaviya Selvam

**Objective:** Sarcasm detection is an essential task in NLP that focuses on identifying the underlying meaning in English text. With a wide variance of perception on the definition of sarcasm among individuals, sarcasm detection is inherently challenging. As such, we aim to develop a robust model that can capture a broad spectrum of sarcastic expressions, rather than solely focusing on specific contexts. Such generalization power of sarcasm detection is crucial for its performance on other applications, one of which is sentiment analysis where the presence or absence of sarcasm could influence the inferred sentiment within a piece of text. 

**Datasets Used:** 1) Sarcasm Corpus V2, 2) Kaggle Reddit Dataset

**Models Explored:** 1) DistilBERT, 2) ALBERT, 3) RoBERTa, 4) XLM-RoBERTa, 5) Ensemble of RoBERTa and XLM-RoBERTa (RobXLMRob), 6) Ensemble of ALBERT, DistilBERt, and RobXLMRob - The final chosen model was model 6. 

**Results:** Test F1 score of 0.796 
