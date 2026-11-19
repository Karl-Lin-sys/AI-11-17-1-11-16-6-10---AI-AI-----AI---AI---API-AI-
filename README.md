# Industrial Inspection Pipeline & AI Job Analysis

This repository contains the production-ready industrial inspection pipeline designed for AOI (Automated Optical Inspection) systems, aligned with top-tier conference architectures (CVPR / NeurIPS) and official framework specifications (PyTorch & OpenCV).

## 1. Architecture & Design Principles

- **Framework Alignment**: Uses `torch.inference_mode()` and zero-copy tensor views to eliminate autograd graph accumulation and memory leaks.
- **Handling Extreme Class Imbalance**: Addresses the structural flaw of manufacturing imbalance through Focal Loss / Balanced Softmax and unsupervised anomaly memory banks (PatchCore / SimpleNet).
- **Leakage Prevention**: Enforces stratified GroupKFold cross-validation by manufacturing lot/run to avoid spatial and temporal sampling bias.

## 2. NIST Cybersecurity Framework (CSF) Alignment

- **Identify (ID)**: Input image validation against arbitrary dimensions and corruption.
- **Protect (PR)**: Execution sandboxing and avoidance of unsafe native deserialization (e.g. `weights_only=True`).
- **Detect (DE)**: Probabilistic anomaly thresholds and confidence calibration for out-of-distribution defects.
- **Respond (RS)**: Real-time exception catching and fallback telemetry logging.
- **Recover (RC)**: Deterministic, stateless worker execution enabling zero-downtime recovery.

## 3. Canadian English Summary (Super Simple for Kids, eh!)

Hey bud! Imagine a busy factory up north making toy wooden beavers. Our helper camera looks at each beaver as fast as a flying hockey puck, brightens up any dark shadows, and checks if anything is broken or missing an ear. When a part looks funny, it flags it right away, cleans up all its toy blocks, and keeps the factory humming smoothly without making the computer tired!
