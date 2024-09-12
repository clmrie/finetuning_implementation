from datasets import load_dataset

from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer, TrainingArguments, Trainer)
from transformers import DataCollatorWithPadding
import evaluate
import torch

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, AdaLoraConfig, IA3Config, PrefixTuningConfig, TaskType
import numpy as np
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from datetime import datetime
import os
from tqdm import tqdm

from transformers import TrainerCallback
from peft import PeftModel
import torch
from safetensors.torch import save_file
import copy


class SavePeftModelCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        
        torch.save(kwargs["model"].state_dict(), os.path.join(checkpoint_folder, "pytorch_model.bin"))
        
        return control

class Finetuning: 

    def __init__(self):
        self.dataset = load_dataset('nyu-mll/multi_nli', split={'train': 'train[:5%]', 'validation_matched': 'validation_matched[:5%]', 'validation_mismatched': 'validation_mismatched[:5%]'})
        self.model_checkpoint = 'bert-base-uncased' 
        self.accuracy = evaluate.load("accuracy")

    def loading_model(self):

        self.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        self.label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_checkpoint, num_labels=3, id2label=self.id2label, label2id=self.label2id)
        
        return model
    
    def preprocess_data(self, model):

        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, add_prefix_space=True)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

        return tokenizer

    def tokenize_function(self, examples, tokenizer):
        tokenizer.truncation_side = "left"
        tokenized_inputs = tokenizer(
            examples['premise'],
            examples['hypothesis'],
            return_tensors="np",
            truncation=True,
            max_length=512
        )
        tokenized_inputs['labels'] = examples['label']
        return tokenized_inputs

    def tokenize_datasets(self, tokenizer):
        return self.dataset.map(lambda examples: self.tokenize_function(examples, tokenizer), batched=True)

    def print_untrained_results(self):
        examples = self.dataset['train'].select(range(5))
        
        print("Untrained model predictions:")
        print("----------------------------")
        for example in examples:
            premise = example['premise']
            hypothesis = example['hypothesis']
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, max_length=512)
            logits = self.model(inputs.input_ids).logits
            preds = torch.argmax(logits, dim=1)
            print(f"Premise: {premise}")
            print(f"Hypothesis: {hypothesis}")
            print(f"Prediction: {self.id2label[preds.item()]}\n")

    def compute_metrics(self, p):
        preds, labels = p
        preds = np.argmax(preds, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    

    def training_model(self):
        print("Starting training with Prefix Tuning.")
        peft_config = PrefixTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=20,
            prefix_projection=True,
            token_dim=768,
            num_layers=12,
            num_attention_heads=12,
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        lr = 1e-4
        batch_size = 32
        num_epochs = 1
        
        training_args = TrainingArguments(
            output_dir="./output",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="steps", 
            save_steps=500,  
            save_total_limit=3,  
            metric_for_best_model="accuracy",
            warmup_ratio=0.1,
            logging_steps=100,
            gradient_accumulation_steps=2,
            lr_scheduler_type="cosine_with_restarts",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation_matched"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[SavePeftModelCallback()],
        )

        trainer.train()
        eval_result = trainer.evaluate()

        return eval_result

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path):
        config = PeftConfig.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path)
        self.model = PeftModel.from_pretrained(self.model, path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    def save_metrics(self, split, metrics):
        with open(f"{split}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)


    def evaluate_model(self, split="validation_mismatched"):
        print("Using the current model state.")

        eval_dataset = self.tokenized_dataset[split]
        
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(output_dir="./eval_output", per_device_eval_batch_size=16),
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        metrics = trainer.evaluate()
        self.save_metrics(f"eval_{split}", metrics)
        return metrics

    def init_training(self):

        self.model = self.loading_model()
        self.tokenizer  = self.preprocess_data(self.model)
        self.tokenized_dataset = self.tokenize_datasets(self.tokenizer)

        self.data_collator = DataCollatorWithPadding(tokenizer = self.tokenizer)

    def evaluate_untrained_model(self):
        print("Evaluating untrained model:")
        print("---------------------------")
        
        untrained_metrics = {}
        
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(output_dir="./untrained_eval_output", per_device_eval_batch_size=16),
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )
        
        for split in ['train', 'validation_matched', 'validation_mismatched']:
            print(f"\nEvaluating {split} split:")
            metrics = trainer.evaluate(eval_dataset=self.tokenized_dataset[split])
            
            metrics = {k.replace('eval_', ''): v for k, v in metrics.items()}
            
            untrained_metrics[split] = metrics
            self.save_metrics(f"untrained_model_{split}", metrics)
            self.print_metrics(metrics)
        
        return untrained_metrics

    def evaluate_trained_model(self):
        print("Evaluating trained model:")
        print("-------------------------")

        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(output_dir="./trained_eval_output", per_device_eval_batch_size=16),
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )

        trained_metrics = {}
        for split in ['train', 'validation_matched', 'validation_mismatched']:
            print(f"\nEvaluating {split} split:")
            metrics = trainer.evaluate(eval_dataset=self.tokenized_dataset[split])
            
            metrics = {k.replace('eval_', ''): v for k, v in metrics.items()}
            
            trained_metrics[split] = metrics
            self.save_metrics(f"trained_model_{split}", metrics)
            self.print_metrics(metrics)

        return trained_metrics

    def compare_metrics(self, untrained_metrics, trained_metrics):
        print("\nComparison of Untrained vs Trained Model Metrics:")
        print("------------------------------------------------")
        for split in ['train', 'validation_matched', 'validation_mismatched']:
            print(f"\n{split.capitalize()} Split:")
            print("Metric      Untrained    Trained     Difference")
            print("-------     ---------    -------     ----------")
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                untrained_value = untrained_metrics[split][metric]
                trained_value = trained_metrics[split][metric]
                difference = trained_value - untrained_value
                print(f"{metric:<12} {untrained_value:.4f}       {trained_value:.4f}      {difference:+.4f}")

    def print_metrics(self, metrics):
        for key, value in metrics.items():
            print(f"{key.capitalize()}: {value:.4f}")

if __name__ == '__main__':
    FT = Finetuning()
    FT.init_training()
    
    untrained_metrics = FT.evaluate_untrained_model()

    FT.training_model()

    FT.save_model("./final_model")

    trained_metrics = FT.evaluate_trained_model()
    
    FT.compare_metrics(untrained_metrics, trained_metrics)

    FT.load_model("./final_model")