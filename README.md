mlops_cookiecutter
==============================

Description and Motivation
--------
This is a ML project on the popular MNIST fashion dataset. 

The motivation behind the project isn't diving deep into NN architectures and optimizing 
given metrics, but more so organizing the project in a correct, formal way and applying 
MLOps techniques.

Some tools, libraries and techniques that were used were:
- Github and DVC for code and data version control.
- Cookiecutter, black, isort, flake8 and Typing for good coding practices.
- Docker and Hydra for reproducibility.
- Pdb for debugging.
- Pytorch and Tensorboard for profiling.
- WandB for experiment logging.
- Pytorch Lightning for minimizing boilerplate.
- Github Actions for Continuous Integration and Continuous Machine Learning.
- Google Cloud Platform, Torchserve and FastAPI for API creation and model deployment locally and on the cloud.

In summary, I developped a NN using Pytorch Lightning and a CI pipeline using github, dvc and docker.
Then, I created a FastAPI app with that model and deployed it at the Google Cloud Platform.

This project follows step by step the guidelines prodived by DTU's course 02476: Machine 
Learning Operations.


How To Run
--------
All commands should be run from the base parent directory MNIST_mlops.

Train the model: python src/models/train_model.py
Test the model on unseen data: python src/models/predict_model.py
Visualize the 2d features of the training dat: python src/visualization/visualize.py

In case anything cannot run and you get an error: "error: no commands supplied", just run: python src/<subdir>/<script.py> install


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
