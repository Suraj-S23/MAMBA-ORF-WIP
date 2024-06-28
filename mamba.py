import argparse
import numpy as np
import pandas as pd
import torch
import time
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer, MambaForCausalLM
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torchmetrics
import os
import random

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, targets, tokenizer, max_length):
        self.sequences = sequences
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.sequences[idx], dtype=torch.long),
            'labels': torch.tensor(self.targets[idx], dtype=torch.float)
        }

class LitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.classifier = torch.nn.Linear(model.config.vocab_size, 1)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_preds = []
        self.test_labels = []

        # Initialize weights properly
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            torch.nn.init.zeros_(self.classifier.bias)

        # Define metrics
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.test_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary')
        self.val_recall = torchmetrics.Recall(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.test_precision = torchmetrics.Precision(task='binary')
        self.test_recall = torchmetrics.Recall(task='binary')
        self.test_f1 = torchmetrics.F1Score(task='binary')

    def forward(self, input_ids, labels):
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits  # Shape: [batch_size, sequence_length, vocab_size]
        logits = self.classifier(logits).squeeze(-1)  # Project logits to [batch_size, sequence_length, 1] and squeeze to [batch_size, sequence_length]
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        logits = self.forward(input_ids, labels)
        loss = self.loss_fn(logits, labels)
        
        # Calculate and log accuracy
        preds = torch.sigmoid(logits) > 0.5
        self.train_accuracy(preds, labels.int())
        self.log('train_accuracy', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Debugging: Print intermediate values
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN or Inf detected in loss at step {batch_idx}")
            print(f"Input IDs: {input_ids}")
            print(f"Logits: {logits}")
            print(f"Labels: {labels}")
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        logits = self.forward(input_ids, labels)
        loss = self.loss_fn(logits, labels)
        
        # Calculate and log metrics
        preds = torch.sigmoid(logits) > 0.5
        self.val_accuracy(preds, labels.int())
        self.val_precision(preds, labels.int())
        self.val_recall(preds, labels.int())
        self.val_f1(preds, labels.int())
        self.log('val_accuracy', self.val_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_precision', self.val_precision, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recall', self.val_recall, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', self.val_f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Debugging: Print intermediate values
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN or Inf detected in validation loss at step {batch_idx}")
            print(f"Input IDs: {input_ids}")
            print(f"Logits: {logits}")
            print(f"Labels: {labels}")
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.append({'val_loss': loss})
        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
            self.log('avg_val_loss', avg_loss, prog_bar=True)
            self.validation_step_outputs.clear()
        # Clear GPU cache
        torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        logits = self.forward(input_ids, labels)
        loss = self.loss_fn(logits, labels)
        
        # Calculate and log metrics
        preds = torch.sigmoid(logits) > 0.5
        self.test_accuracy(preds, labels.int())
        self.test_precision(preds, labels.int())
        self.test_recall(preds, labels.int())
        self.test_f1(preds, labels.int())
        self.log('test_accuracy', self.test_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_precision', self.test_precision, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_recall', self.test_recall, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_f1', self.test_f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Store predictions and labels for analysis
        self.test_preds.extend(preds.cpu().numpy().flatten())
        self.test_labels.extend(labels.cpu().numpy().flatten())
        
        # Debugging: Print intermediate values
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN or Inf detected in test loss at step {batch_idx}")
            print(f"Input IDs: {input_ids}")
            print(f"Logits: {logits}")
            print(f"Labels: {labels}")
        
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.test_step_outputs.append({'test_loss': loss})
        return {'test_loss': loss}

    def on_test_epoch_end(self):
        if self.test_step_outputs:
            avg_loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean()
            self.log('avg_test_loss', avg_loss, prog_bar=True)
            self.test_step_outputs.clear()
        # Clear GPU cache
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.00001)  # Further reduced learning rate

def calculate_metrics(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            # Placeholder for labels, replace with actual labels if needed
            labels = torch.zeros_like(input_ids[:, :1]).to(device)
            logits = model(input_ids, labels)  # Adjusted to include labels
            
            # Convert logits to probabilities
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()  # Binary classification threshold
            
            # Calculate metrics here
            accuracy = (preds == labels).sum().item() / len(preds)
            metrics['inference_accuracy'] = accuracy
    
    return metrics

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")

    # Load data from Parquet files
    train_data_path = args.input + 'train_data.parquet'
    inference_data_path = args.input + 'inference_data.parquet'

    train_df = pd.read_parquet(train_data_path)
    inference_df = pd.read_parquet(inference_data_path)

    # Splitting the training data into training, validation, and testing sets
    train_df_split = train_df.sample(frac=0.75, random_state=42)  # 75% of the data for training
    temp_df = train_df.drop(train_df_split.index)
    valid_df = temp_df.sample(frac=0.6, random_state=42)  # 60% of the remaining data for validation
    test_df = temp_df.drop(valid_df.index)  # Remaining 40% for testing

    # Tokenizing the sequences
    train_tokenized_sequences = [tokenizer.encode(seq, max_length=5000, truncation=True, padding="max_length") for seq in train_df_split['sequence']]
    valid_tokenized_sequences = [tokenizer.encode(seq, max_length=5000, truncation=True, padding="max_length") for seq in valid_df['sequence']]
    test_tokenized_sequences = [tokenizer.encode(seq, max_length=5000, truncation=True, padding="max_length") for seq in test_df['sequence']]
    inference_tokenized_sequences = [tokenizer.encode(seq, max_length=5000, truncation=True, padding="max_length") for seq in inference_df['sequence']]

    # Assuming 'targets' column exists and contains float values for binary classification
    train_targets = np.array([np.array(target).astype(float) for target in train_df_split['target'].tolist()])
    valid_targets = np.array([np.array(target).astype(float) for target in valid_df['target'].tolist()])
    test_targets = np.array([np.array(target).astype(float) for target in test_df['target'].tolist()])
    inference_targets = np.array([np.array(target).astype(float) for target in inference_df['target'].tolist()])

    # Create datasets
    train_dataset = SequenceDataset(train_tokenized_sequences, train_targets, tokenizer, max_length=5000)
    valid_dataset = SequenceDataset(valid_tokenized_sequences, valid_targets, tokenizer, max_length=5000)
    test_dataset = SequenceDataset(test_tokenized_sequences, test_targets, tokenizer, max_length=5000)
    inference_dataset = SequenceDataset(inference_tokenized_sequences, inference_targets, tokenizer, max_length=5000)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)  # Shuffle is False for validation
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)  # Shuffle is False for testing
    inference_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=False)  # Shuffle is False for inference

    lit_model = LitModel(model)
    early_stopping = EarlyStopping(
        monitor='train_loss',
        min_delta=0,
        patience=5,
        verbose=True,
        mode='min'
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision=16,
        accumulate_grad_batches=4,
        gradient_clip_val=1.0,
        callbacks=[
            ModelCheckpoint(dirpath='./model/', monitor='train_loss'),
            early_stopping
        ])
    trainer.fit(lit_model, train_loader, valid_loader)  # Now including valid_loader for validation

    start_inference_time = time.time()
    trainer.test(lit_model, test_loader)  # Using test_loader for testing

    # Assuming you have a way to calculate metrics based on the model's output
    # For demonstration, let's assume you have a function `calculate_metrics` that takes model's output and returns metrics dictionary
    metrics = calculate_metrics(lit_model, inference_loader, device)

    # Log the metrics
    for metric_name, metric_value in metrics.items():
        lit_model.log(metric_name, metric_value, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    end_inference_time = time.time()
    print(f"Inference completed in {end_inference_time - start_inference_time} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Mamba model on preprocessed DNA sequences.')
    parser.add_argument('--input', type=str, required=True, help='Base name for Parquet files (e.g., "data" for "data_train.parquet" and "data_inference.parquet").')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training.')
    args = parser.parse_args()
    main(args)