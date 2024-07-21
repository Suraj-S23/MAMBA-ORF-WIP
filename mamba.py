import argparse
import numpy as np
import pandas as pd
import torch
import time
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torchmetrics
import os
import yaml
from peft import LoraConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import wandb

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, targets, orf_lengths, tokenizer, max_length):
        self.sequences = sequences
        self.targets = targets
        self.orf_lengths = orf_lengths
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.sequences[idx], dtype=torch.long),
            'labels': torch.tensor(self.targets[idx], dtype=torch.float),
            'orf_lengths': torch.tensor(self.orf_lengths[idx], dtype=torch.long)
        }

class LitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.classifier = torch.nn.Linear(model.config.vocab_size, 1)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.validation_step_outputs = []
        self.test_step_outputs = []  # Ensure this is initialized

        torch.nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            torch.nn.init.zeros_(self.classifier.bias)

        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.test_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary')
        self.val_recall = torchmetrics.Recall(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.test_precision = torchmetrics.Precision(task='binary')
        self.test_recall = torchmetrics.Recall(task='binary')
        self.test_f1 = torchmetrics.F1Score(task='binary')

    def forward(self, input_ids, labels=None):
        outputs = self.model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits
        logits = self.classifier(logits).squeeze(-1)
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return logits, loss
        else:
            return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        output = self(input_ids, labels)
        logits, loss = output if isinstance(output, tuple) else (output, None)
        preds = torch.sigmoid(logits) > 0.5
        self.train_accuracy(preds, labels.int())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        output = self(input_ids, labels)
        logits, loss = output if isinstance(output, tuple) else (output, None)
        preds = torch.sigmoid(logits) > 0.5
        self.val_accuracy(preds, labels.int())
        self.val_precision(preds, labels.int())
        self.val_recall(preds, labels.int())
        self.val_f1(preds, labels.int())
        self.log('val_accuracy', self.val_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_precision', self.val_precision, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recall', self.val_recall, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', self.val_f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if loss is not None:
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.validation_step_outputs.append(loss.detach())  # Collect loss for averaging
        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.validation_step_outputs).mean() if self.validation_step_outputs else torch.tensor(0.0)
        self.log('avg_val_loss', avg_val_loss, prog_bar=True)
        self.validation_step_outputs.clear()  # Clear for the next epoch
        torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        output = self(input_ids, labels)
        logits, loss = output if isinstance(output, tuple) else (output, None)
        preds = torch.sigmoid(logits) > 0.5
        self.test_accuracy(preds, labels.int())
        self.test_precision(preds, labels.int())
        self.test_recall(preds, labels.int())
        self.test_f1(preds, labels.int())
        self.log('test_accuracy', self.test_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_precision', self.test_precision, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_recall', self.test_recall, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_f1', self.test_f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if loss is not None:
            self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.test_step_outputs.append(loss.detach())  # Collect loss for averaging
        return {'test_loss': loss}

    def on_test_epoch_end(self):
        avg_test_loss = torch.stack(self.test_step_outputs).mean() if self.test_step_outputs else torch.tensor(0.0)
        self.log('avg_test_loss', avg_test_loss, prog_bar=True)
        self.test_step_outputs.clear()  # Clear for the next epoch
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)  # Reduced learning rate and changed optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

class LengthBasedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, max_orf_length=None):
        if max_orf_length is None:
            self.indices = list(range(len(dataset)))
        else:
            # Adjusted to filter based on any ORF < 150 in a sequence
            self.indices = [i for i, lengths in enumerate(dataset.orf_lengths) if any(length <= max_orf_length for length in lengths)]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def load_data(input_path):
    print("Loading data from:", input_path)
    train_data_path = os.path.join(input_path, 'train_data.parquet')
    inference_data_path = os.path.join(input_path, 'inference_data.parquet')
    try:
        train_df = pd.read_parquet(train_data_path)
        inference_df = pd.read_parquet(inference_data_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    return train_df, inference_df

def load_archaea_data(archaea_data_path):
    print("Loading Archaea data from:", archaea_data_path)
    inference_data_path = os.path.join(archaea_data_path, 'inference_data.parquet')
    try:
        inference_df = pd.read_parquet(inference_data_path)
        print("Archaea data loaded successfully.")
    except Exception as e:
        print(f"Error loading Archaea data: {e}")
        raise
    return inference_df

def split_data(train_df):
    print("Splitting data...")
    train_df_split = train_df.sample(frac=0.75, random_state=42)
    temp_df = train_df.drop(train_df_split.index)
    valid_df = temp_df.sample(frac=0.6, random_state=42)
    test_df = temp_df.drop(valid_df.index)
    print("Data split successfully.")
    return train_df_split, valid_df, test_df

def tokenize_sequences(tokenizer, df, max_length=5000):
    print("Tokenizing sequences...")
    tokenized_sequences = [tokenizer.encode(seq, max_length=max_length, truncation=True, padding="max_length") for seq in df['sequence']]
    print("Sequences tokenized successfully.")
    return tokenized_sequences

def create_datasets(tokenizer, train_df_split, valid_df, test_df, inference_df):
    print("Creating datasets...")
    train_tokenized_sequences = tokenize_sequences(tokenizer, train_df_split)
    valid_tokenized_sequences = tokenize_sequences(tokenizer, valid_df)
    test_tokenized_sequences = tokenize_sequences(tokenizer, test_df)
    inference_tokenized_sequences = tokenize_sequences(tokenizer, inference_df)

    train_targets = np.array([np.array(target).astype(float) for target in train_df_split['target'].tolist()])
    valid_targets = np.array([np.array(target).astype(float) for target in valid_df['target'].tolist()])
    test_targets = np.array([np.array(target).astype(float) for target in test_df['target'].tolist()])
    inference_targets = np.array([np.array(target).astype(float) for target in inference_df['target'].tolist()])

    train_dataset = SequenceDataset(train_tokenized_sequences, train_targets, train_df_split['orf_lengths'].tolist(), tokenizer, max_length=5000)
    valid_dataset = SequenceDataset(valid_tokenized_sequences, valid_targets, valid_df['orf_lengths'].tolist(), tokenizer, max_length=5000)
    test_dataset = SequenceDataset(test_tokenized_sequences, test_targets, test_df['orf_lengths'].tolist(), tokenizer, max_length=5000)
    inference_dataset = SequenceDataset(inference_tokenized_sequences, inference_targets, inference_df['orf_lengths'].tolist(), tokenizer, max_length=5000)

    print("Datasets created successfully.")
    return train_dataset, valid_dataset, test_dataset, inference_dataset

def create_archaea_dataset(tokenizer, archaea_df, max_length=5000):
    print("Creating Archaea dataset...")
    archaea_tokenized_sequences = tokenize_sequences(tokenizer, archaea_df)
    archaea_targets = np.array([np.array(target).astype(float) for target in archaea_df['target'].tolist()])
    archaea_dataset = SequenceDataset(archaea_tokenized_sequences, archaea_targets, archaea_df['orf_lengths'].tolist(), tokenizer, max_length=5000)
    print("Archaea dataset created successfully.")
    return archaea_dataset

def load_config(config_path):
    print("Loading configuration from:", config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print("Configuration loaded successfully.")
    return config

def main(config):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading tokenizer and model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
        model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf").to(device)
        print("Tokenizer and model loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer or model: {e}")
        raise

    # Apply LoRA to the model
    print("Applying LoRA to the model...")
    lora_config = LoraConfig(
        r=8,  # Rank of the low-rank adaptation
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.1,  # Dropout rate
        target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],  # Correct target modules
        task_type="CAUSAL_LM",
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    print("LoRA applied successfully.")

    lit_model = LitModel(model).to(device)

    print("Loading data...")
    train_df, inference_df = load_data(config['input_path'])
    train_df_split, valid_df, test_df = split_data(train_df)
    train_dataset, valid_dataset, test_dataset, inference_dataset = create_datasets(tokenizer, train_df_split, valid_df, test_df, inference_df)

    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)
    inference_loader = DataLoader(inference_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)

    small_orfs_sampler = LengthBasedSampler(test_dataset, max_orf_length=150)
    small_orfs_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, sampler=small_orfs_sampler, num_workers=8, pin_memory=True)

    # Load and prepare Archaea data
    print("Loading and preparing Archaea data...")
    archaea_df = load_archaea_data(config['archaea_data_path'])
    archaea_dataset = create_archaea_dataset(tokenizer, archaea_df)
    archaea_loader = DataLoader(archaea_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)
    print("Archaea data prepared successfully.")

    # Initialize W&B
    print("Initializing W&B...")
    wandb_logger = WandbLogger(project=config['wandb_project'], entity=config.get('wandb_entity', None), name=config['log_name'])
    print("W&B initialized.")

    logger = TensorBoardLogger("tb_logs", name=config['log_name'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=True, mode='min')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=config['checkpoint_path'], filename='best-checkpoint', save_top_k=1, mode='min')

    print("Initializing PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        logger=[logger, wandb_logger],
        callbacks=[checkpoint_callback, early_stopping],
        max_epochs=config['epochs'],
        precision=16,  # Use mixed precision training
        accumulate_grad_batches=4,
        gradient_clip_val=1.0,  # Added gradient clipping
        accelerator='gpu',  # Use 'gpu' for GPU training
        devices=1  # Number of GPUs to use
    )
    print("Trainer initialized successfully.")

    print("Starting training...")
    trainer.fit(lit_model, train_loader, valid_loader)
    print("Training completed.")

    print("Evaluating on all ORFs...")
    trainer.test(lit_model, test_loader)
    print("Evaluation on all ORFs completed.")

    print("Evaluating on small ORFs (<150 bases)...")
    trainer.test(lit_model, small_orfs_loader)
    print("Evaluation on small ORFs completed.")

    print("Evaluating on Archaea data...")
    trainer.test(lit_model, archaea_loader)
    print("Evaluation on Archaea data completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Mamba model on preprocessed DNA sequences.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)

