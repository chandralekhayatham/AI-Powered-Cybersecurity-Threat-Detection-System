# Architecture

```text
Synthetic network telemetry generator / CSV
              |
              v
Pandas validation and feature selection
              |
              v
Scikit-learn preprocessing pipeline
(numeric imputation + protocol one-hot encoding)
              |
              v
Random Forest classifier
              |
      +-------+--------+
      |                |
      v                v
Evaluation metrics    Prediction + confidence
      |                |
      v                v
Metrics/charts        Review-level alert
      |                |
      +------- Flask API + local dashboard
```

The project is an offline educational simulation. The classifier learns patterns from generated numeric features and a protocol category. It does not collect packets, scan hosts, or establish outbound connections. `model_confidence` is the maximum class probability, not a calibrated guarantee of correctness.
