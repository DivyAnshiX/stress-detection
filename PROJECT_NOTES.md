# Project notes

## First working direction

The existing notebook proved that the dataset can support binary text classification, but it had repeated cells, interactive `input()` calls, and several heavyweight experiments mixed together. The first application version uses one reproducible TF-IDF + logistic regression pipeline.

## Decisions

- Keep the original notebook as reference instead of deleting the exploration.
- Use a stratified 80/20 split so both labels remain represented in evaluation.
- Save the vectorizer and classifier together through a scikit-learn pipeline.
- Use probability wording in the UI, not certainty wording.
- Add a clear non-diagnostic disclaimer because this is a student ML project.

## Current environment note

The active Windows device has the project files and the Python launcher, but no installed Python interpreter was available during setup. The code is prepared and should be run after installing Python 3.10+ and the packages in `requirements.txt`.

## Next debugging pass

1. Run `python train_model.py` and record the actual metrics in `model/metrics.json`.
2. Try neutral, stressed, sarcastic, and very short example texts.
3. Check whether label errors are concentrated in one class.
4. Update the README with the measured accuracy, precision, recall, and F1 score rather than guessing them.
