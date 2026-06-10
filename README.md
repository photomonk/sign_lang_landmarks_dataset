# Sign Language Landmark Dataset

## Overview

This dataset contains hand landmark coordinates extracted from sign language videos using MediaPipe Hands. The dataset is designed for training and evaluating machine learning models for sign language recognition and translation.

## Contents

The dataset includes:

- English Alphabet Signs (A–Z)
- Common Word Signs
- Landmark coordinates stored as NumPy arrays (`.npy`)
- Hand skeleton data consisting of 21 hand landmarks per frame

## Data Format

Each sample is stored as a NumPy file.

```python
[
    [x1, y1, z1],
    [x2, y2, z2],
    ...
    [x21, y21, z21]
]

Raw source :
aphabets:https://youtu.be/_YDrZ8d5SRk?si=eNAzCNPCxO5mXCBJ
words:https://youtu.be/VPEnQ-k_Fbg?si=Erjai9NN7rzBP9QF
