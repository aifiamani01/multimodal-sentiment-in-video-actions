# Multimodal Emotion Recognition

## Motivation
Understanding human emotions from multiple modalities (video, facial expressions, and audio) is a critical challenge in AI.  
Most emotion recognition systems focus on a single modality, but combining multiple modalities can improve robustness and capture subtle emotional cues.  

This project aims to explore how multimodal representations can be leveraged for emotion recognition, starting with analysis of frozen pretrained models.

---

## Core Research Questions
1. Can multimodal representations (video + facial + audio) improve emotion recognition compared to single modalities?  
2. Which modalities contribute most to recognizing specific emotions?  
3. How does temporal information in video and audio affect recognition performance?  
4. Can frozen pretrained models provide embeddings that capture emotional content without fine-tuning?  
5. What are the limitations and failure cases of multimodal emotion recognition in small datasets?

---

## Scope
- Analysis-focused: initial experiments will use frozen pretrained models for each modality.  
- Proof-of-concept: small datasets (e.g., RAVDESS) will be used initially.  
- Not aiming for SOTA at this stage; focus is on understanding embeddings, modality contributions, and research workflow.

---

## Planned Approach
1. Extract embeddings from each modality:
   - **Video:** pretrained TimeSformer for action recognition  
   - **Face:** pretrained face recognition / expression model  
   - **Audio:** pretrained speech / audio model (e.g., Wav2Vec2, YAMNet)  
2. Train a simple classifier on concatenated embeddings (linear probe / small MLP)  
3. Visualize and analyze embeddings using PCA/t-SNE  
4. Identify which modalities contribute most to emotion recognition  
5. Document preliminary observations, limitations, and open questions

---


## Repository Structure
experiments/ # Scripts for embedding extraction and classifier training
analysis/ # Jupyter notebooks for visualizations and evaluations
data/ # Raw and processed dataset files
README.md # This file
research_questions.md # Guiding research questions
data_notes.md # Notes about dataset sources, preprocessing, and labels
limitations.md # Known limitations and challenges


---

## Status
- Repository created  
- Initial planning, research questions, and dataset notes documented  
- Next step: implement embedding extraction for each modality

## First Commit
Repository setup completed. Ready for experiments.


