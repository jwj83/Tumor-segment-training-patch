# Tumor-segment training patch

This repository contains the two updated training entry points for the cloud training environment.

Copy these files over the existing checkout while preserving the `training/` path:

- `training/train_medicalnet.py`
- `training/train_goal4.py`
- `training/models/medicalnet.py`

The adapter supports the official MedicalNet `models.resnet.resnet50` API and classifier-style forks. Both training scripts accept the optional `--medicalnet-checkpoint` argument. Omitting it keeps the original random-initialization behavior.

The patch was checked with Python compilation, both `--help` commands, and the existing test suite in the source repository.
