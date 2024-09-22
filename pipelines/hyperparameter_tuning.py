# pipelines/hyperparameter_tuning.py

import sys
import os
import optuna
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import DataLoader
import logging
import yaml

# Add the project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(
    filename="hyperparameter_tuning.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("hyperparameter_tuning")


def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.
    """
    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Hyperparameters to tune
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-4)
    num_train_epochs = trial.suggest_int("num_train_epochs", 2, 5)
    per_device_train_batch_size = trial.suggest_categorical(
        "per_device_train_batch_size", [16, 32]
    )

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
    )

    from data_loader import load_datasets  # Ensure this function exists

    train_ds, eval_ds = load_datasets()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result["eval_loss"]


def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    logger.info("Best hyperparameters:")
    logger.info(study.best_params)
    logger.info(f"Best evaluation loss: {study.best_value}")


if __name__ == "__main__":
    main()
