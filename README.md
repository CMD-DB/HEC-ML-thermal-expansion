Code author: Insu Kim (isbh4107@gmail.com)

# HEC machine learning and thermal expansion

Composition-based SVR and QHA-calibrated thermal-expansion analysis of high-entropy carbides.

## Run

Python 3.12.3.

```sh
python -m pip install -r requirements.txt
python prepare_data.py
python train_models.py
python thermal.py
```

## Files

- `prepare_data.py`: data cleaning and 12-descriptor construction.
- `train_models.py`: SVR training and nested family-excluded validation.
- `thermal.py`: thermal expansion and ROM/Turner/Kerner evaluation.
- `data/`: numerical inputs.
- `config.json`: hyperparameters, random seed and sampling settings.

Outputs are saved in `results/`.
