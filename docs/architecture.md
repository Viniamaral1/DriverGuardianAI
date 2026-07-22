# DriverGuardianAI Architecture

This document describes the overall architecture of DriverGuardianAI.

## Pipeline

Raw Video
    ?
MediaPipe Face Mesh
    ?
Feature Extraction
    ?
Machine Learning Model
    ?
Temporal Decision Engine
    ?
Driver Alert

The V2 pipeline consists of:

- Face landmark detection
- Eye Aspect Ratio (EAR)
- Yawn detection
- Head pose estimation
- Feature engineering
- Histogram Gradient Boosting classifier
- Confidence-aware temporal rule engine
- Explainability (SHAP)
