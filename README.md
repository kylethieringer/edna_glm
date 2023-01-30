# edna_glm

## preprocessing
- edit experiment paths in `main()` in `features.py` and `song_utils.py` 
- run in terminal with 
    ```
    conda activate env
    python features.py
    python song_utils.py
    ```
- the outputs of these files are hdf5 files with kinematic behavior features or song features

## GLM
- `glm.py`
  - decide stimulus history window in frames
  - change experiment paths and savepaths in `main()`
  - run in terminal with 
    ```
    conda activate env
    python glm.py
    ```
- the output is png files of the filters and a results.csv containing:
  - pCor = percent correct
  - logloss
  - filter_norms
  - f1_score
  - deviance (calculated by `2*sklearn.metrics.log_loss(y, model.predict_proba(x), normalize=False)`)