# Tumor-segment training patch

This patch updates the MedicalNet training path for the competition rules.

Files to copy into the existing training checkout:

- `training/train_medicalnet.py`
- `training/train_goal4.py`
- `training/models/medicalnet.py`

The classification and Goal4 training entries now use one `AccessionNumber` per sample with up to four modality channels (T1, T1CE, T2, FLAIR), zero filling missing modalities. They support the official MedicalNet `models.resnet.resnet50` API, optional pretrained checkpoints, four data-loader workers by default, 64x64x32 inputs, class weighting, ReduceLROnPlateau learning-rate adaptation, early stopping, and `last.pt` resume checkpoints.

Example optional arguments:

```text
--medicalnet-checkpoint /path/to/medicalnet_weight.pth
--num-workers 4
--target-shape 64 64 32
--lr 1e-4
--patience 12
```
