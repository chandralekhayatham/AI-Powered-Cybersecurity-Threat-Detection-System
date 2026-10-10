# Safety and limitations

- The included dataset is generated synthetic telemetry and is labeled `SYNTHETIC_DEMO_ONLY`.
- The application binds to `127.0.0.1` and does not capture live network traffic.
- No exploit, malware, credential attack, scanning, or external target activity is included. Attack-like labels are statistical patterns for a classifier demonstration.
- A predicted suspicious label is not proof of compromise; analysts should validate against additional evidence.
- Evaluation results are only for this synthetic dataset. Do not present them as real-world security efficacy.
- Do not expose the demo server to the public internet or send real sensitive logs to this app.
