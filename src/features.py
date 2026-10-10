"""Feature definitions shared by training and prediction."""
NUMERIC_FEATURES = [
    "duration_seconds",
    "src_bytes",
    "dst_bytes",
    "failed_logins",
    "request_frequency",
    "connection_count",
    "packet_size",
    "unique_ports_contacted",
]
CATEGORICAL_FEATURES = ["protocol"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "label"
