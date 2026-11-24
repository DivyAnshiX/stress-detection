# Backend improvement plan

## Completed in this phase

- Added `backend.py` so prediction logic is separate from the UI.
- Added input validation and whitespace normalization.
- Added an ambiguity band between 40% and 60% stress probability.
- Added cautious indication levels instead of treating probability as a diagnosis.
- Added a `needs_more_context` flag for very short or uncertain text.
- Added training-only cross-validation for model selection.
- Added a Complement Naive Bayes comparison against Logistic Regression.
- Kept the final test set untouched until after model selection.
- Saved model comparison, split details, class meanings, and warnings in `model/metrics.json`.

## Why this is better

The first version selected one model directly on the test result and only saved a few metrics. The new version compares candidates using the training partition, selects the better macro-F1 model, and then evaluates once on the held-out test partition. This gives a cleaner estimate of how the model may behave on unseen text.

## Next backend checks

1. Train again with `python train_model.py`.
2. Run a small prediction smoke test through `backend.py`.
3. Review false positives and false negatives from the test set.
4. Add a manually checked validation sample that is separate from `stress.csv`.
5. Only after these checks, update the UI to display the richer prediction object.
