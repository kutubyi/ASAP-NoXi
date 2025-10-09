# ASAP-NoXi

**Fork of ASAP model for NoXi dataset experiments**

Original paper: "ASAP: Endowing Adaptation Capability to Agent in Human-Agent Interaction" [IUI 2023]
https://dl.acm.org/doi/abs/10.1145/3581641.3584081

## This Fork

This repository applies the ASAP model to the [NoXi (Novice-eXpert interaction) dataset](https://multimediate-challenge.org/datasets/Dataset_NoXi/).

**NoXi Dataset Paper:** [Cafaro et al., 2017](https://dl.acm.org/doi/10.1145/3136755.3136780)

**Additions:**
- NoXi dataset preprocessing scripts (`/preprocessing`)
- Feature extraction guides (OpenFace + OpenSMILE)
- Experiment tracking

**Status:** In progress - awaiting NoXi dataset access

---

# Original ASAP Model

Code of Augmented Self-Attention Pruning (ASAP) model, IUI 2023.

## Description of ASAP
In human-human interaction, interlocutors adapt their behaviors reciprocally and dynamically. ASAP models this adaptation mechanism also referred to as reciprocal adaptation between a Socially Interactive Agent (SIA) and a human interlocutor.\
The SIA behavior as speaker and listener is fully driven by ASAP. Only its head and upper facial expressions are computed. The voice of the SIA is dubbed from the original video of the human-human data.

## Demo video
A dyadic interaction between a human and a SIA simulated with the GRETA platform. SIA behaviors are generated via our ASAP model at the frame level at 25 fps.

[![ASAP DEMO](https://user-images.githubusercontent.com/44306168/213715354-b1742b06-8df2-45fc-a01c-91dce49e44c6.png)](http://www.youtube.com/watch?v=feojlOrFCIg "ASAP Demo")
Please click to see the full demo video.

## Requirements
- Python 3.9.7
- Tensorflow 2.4.1
- numpy 1.23.5
- pandas 1.2.0
- scipy 1.4.1
- pydtw 2.0.2

## Instructions
It is possible to train, predict, and evaluate objectively the ASAP model. You can run the code as the following.

To train the ASAP model, run with:
```
python train_ASAP.py
```

To predict or evaluate the ASAP model, run with:
```
python predNeval_ASAP.py
```
You can choose wheither to do a prediction, evaluation or both by selecting the corresponding mode:
- "pred": predict with ASAP
- "eval": evaluate objectively by loading precomputed predictions of ASAP
- "predNeval": predict and evaluate  objectively the predictions of ASAP

