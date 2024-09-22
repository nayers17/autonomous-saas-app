# pipelines/mixed_precision_training.py

import sys
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import logging

# Add the project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(
    filename="mixed_precision_training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mixed_precision_training")


def mixed_precision_train(
    model_name: str, train_dataset, eval_dataset, epochs: int = 3
):
    """
    Trains the model using mixed precision.

    Args:
        model_name (str): Pre-trained model name.
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset.
        epochs (int): Number of training epochs.
    """
    logger.info(f"Loading model '{model_name}' for mixed precision training.")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, num_workers=4)

    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = tokenizer(
                batch["text"], padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(**inputs)
                loss = loss_fn(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_train_loss}")

        # Evaluation
        model.eval()
        total_eval_loss = 0
        correct_predictions = 0
        with torch.no_grad():
            for batch in eval_loader:
                inputs = tokenizer(
                    batch["text"], padding=True, truncation=True, return_tensors="pt"
                ).to(device)
                labels = batch["labels"].to(device)

                with autocast():
                    outputs = model(**inputs)
                    loss = loss_fn(outputs.logits, labels)

                total_eval_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                correct_predictions += torch.sum(preds == labels)

        avg_eval_loss = total_eval_loss / len(eval_loader)
        accuracy = correct_predictions.double() / len(eval_loader.dataset)
        logger.info(
            f"Epoch {epoch+1}/{epochs} - Evaluation Loss: {avg_eval_loss} - Accuracy: {accuracy}"
        )

    # Save the trained model
    model.save_pretrained("models/trained_mixed_precision_model")
    tokenizer.save_pretrained("models/trained_mixed_precision_model")
    logger.info("Mixed precision training completed and model saved.")


if __name__ == "__main__":
    # Example usage
    from data_loader import load_datasets  # Ensure this function exists

    train_ds, eval_ds = load_datasets()
    mixed_precision_train("distilbert-base-uncased", train_ds, eval_ds, epochs=3)
