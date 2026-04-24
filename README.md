# CSCI 434 Final Project

This repo is set up for iterating on Wireshark capture features and training a website-traffic classifier.

## Current dataset

The raw captures live in [`Wireshark Captures`](./Wireshark%20Captures). Right now there is one exported CSV and one `.pcapng` per website label:

- `chatgpt`
- `github`
- `instagram`
- `nbcnews`
- `walmart`

## Workflow

Use the notebook for exploration, but keep reusable logic in Python modules.

1. Create the environment: `uv sync`
2. Start the notebook server: `uv run jupyter notebook`
3. Open `notebooks/feature_exploration.ipynb`
4. Run the repeatable experiment script when you want a clean text summary: `uv run python scripts/run_experiment.py`

The notebook currently:

- loads all capture CSVs
- converts each capture into fixed-size packet windows
- computes starter statistical features
- holds out the newest capture session from each label for testing
- compares a baseline, logistic regression, and random forest
- surfaces the most important random-forest features

## Important modeling caution

Do not treat individual packets as independent training samples. That leaks structure from the same browsing session into both train and test splits.

The starter code uses packet windows as samples, but evaluation is now done by holding out entire capture sessions. If you only have one capture session per website, this split is impossible, so keep collecting multiple sessions per label.
