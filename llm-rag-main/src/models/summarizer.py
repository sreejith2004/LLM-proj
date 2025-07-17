"""
Legal document summarization model with LoRA fine-tuning.
"""

import torch
import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class LegalSummarizer:
    """
    Legal document summarization model with LoRA fine-tuning capabilities.
    """

    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config = self._load_config(config_path)
        self.tokenizer = None
        self.model = None
        self.peft_model = None

    def _load_config(self, config_path: str) -> Dict:
        """Load model configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_base_model(self, model_name: Optional[str] = None):
        """Load the base model and tokenizer."""
        model_name = model_name or self.config['model']['base_model']

        logger.info(f"Loading base model: {model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Auto-detect device and set appropriate settings
        device_kwargs = {}

        if torch.cuda.is_available():
            # CUDA GPU - use quantization if specified
            if self.config.get('quantization', {}).get('load_in_4bit', False):
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                device_kwargs['quantization_config'] = bnb_config
                device_kwargs['device_map'] = "auto"
            device_kwargs['torch_dtype'] = torch.float16

        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            # Apple Silicon - use float32 for better compatibility
            device_kwargs['torch_dtype'] = torch.float32
            # Force CPU for very limited memory scenarios
            try:
                # Test MPS availability with a small tensor
                test_tensor = torch.randn(1, 1).to('mps')
                del test_tensor
            except Exception as e:
                logger.warning(f"MPS test failed: {e}. Falling back to CPU.")
                device_kwargs['torch_dtype'] = torch.float32

        else:
            # CPU - use float32
            device_kwargs['torch_dtype'] = torch.float32

        logger.info(f"Loading model with dtype: {device_kwargs.get('torch_dtype', 'default')}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **device_kwargs
        )

        logger.info("Base model loaded successfully")

    def setup_lora(self):
        """Setup LoRA configuration for efficient fine-tuning."""
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias'],
            task_type=TaskType.CAUSAL_LM,
        )

        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()

        logger.info("LoRA configuration applied")

    def prepare_dataset(self, data_path: str, split: str = "train") -> Dataset:
        """Prepare dataset for training."""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create prompts
        prompts = []
        for item in data:
            prompt = self._create_training_prompt(item['input'], item['target'])
            prompts.append(prompt)

        # Adjust max_length based on device for memory efficiency
        max_length = self.config['model']['max_length']
        if not torch.cuda.is_available():
            max_length = min(max_length, 512)  # Reduce for Mac/CPU
            logger.info(f"Reduced max_length to {max_length} for memory efficiency")

        # Tokenize
        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # Create dataset
        dataset = Dataset.from_dict({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids'].clone()  # For causal LM
        })

        return dataset

    def _create_training_prompt(self, judgment: str, summary: str) -> str:
        """Create training prompt format."""
        return f"""### Legal Document Summarization

**Instruction:** Summarize the following legal judgment concisely, highlighting the key facts, legal issues, reasoning, and conclusion.

**Judgment:**
{judgment}

**Summary:**
{summary}{self.tokenizer.eos_token}"""

    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None, training_config: Optional[Dict] = None):
        """Fine-tune the model using LoRA."""
        if self.peft_model is None:
            raise ValueError("LoRA model not initialized. Call setup_lora() first.")

        # Use provided training config or default
        if training_config is None:
            # Try to load from default training config file
            import yaml
            with open("config/training_config.yaml", 'r') as f:
                training_config = yaml.safe_load(f)

        # Auto-detect device and set appropriate precision
        device = torch.cuda.is_available()
        use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()

        # Adjust batch sizes based on device for memory efficiency
        if device:  # CUDA GPU - use original batch sizes
            use_fp16 = True
            use_bf16 = False
            device_str = "cuda"
            # Keep original batch sizes from config
        else:  # MPS or CPU - reduce memory usage
            use_fp16 = False
            use_bf16 = False
            # Try MPS first, fallback to CPU if needed
            if use_mps:
                device_str = "mps"
            else:
                device_str = "cpu"

            # Reduce batch sizes for limited memory
            if training_config['training']['per_device_train_batch_size'] > 1:
                training_config['training']['per_device_train_batch_size'] = 1
                training_config['training']['per_device_eval_batch_size'] = 1
                training_config['training']['gradient_accumulation_steps'] = 8
                logger.info("Reduced batch sizes for memory efficiency on Mac/CPU")

        logger.info(f"Training device: {'CUDA' if device else 'MPS' if use_mps else 'CPU'}")
        logger.info(f"Using precision: {'fp16' if use_fp16 else 'bf16' if use_bf16 else 'fp32'}")

        # Training arguments with type safety
        training_args = TrainingArguments(
            output_dir=training_config['training']['output_dir'],
            num_train_epochs=int(training_config['training']['num_train_epochs']),
            per_device_train_batch_size=int(training_config['training']['per_device_train_batch_size']),
            per_device_eval_batch_size=int(training_config['training']['per_device_eval_batch_size']),
            gradient_accumulation_steps=int(training_config['training']['gradient_accumulation_steps']),
            learning_rate=float(training_config['training']['learning_rate']),
            weight_decay=float(training_config['training']['weight_decay']),
            warmup_steps=int(training_config['training']['warmup_steps']),
            logging_steps=int(training_config['training']['logging_steps']),
            save_steps=int(training_config['training']['save_steps']),
            eval_steps=int(training_config['training']['eval_steps']),
            eval_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            fp16=use_fp16,
            bf16=use_bf16,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            processing_class=self.tokenizer,  # Updated from tokenizer to processing_class
        )

        # Start training
        logger.info("Starting training...")
        trainer.train()

        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)

        logger.info(f"Training completed. Model saved to {training_args.output_dir}")

    def summarize(self, judgment: str, max_new_tokens: int = None) -> str:
        """Generate summary for a legal judgment."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_base_model() first.")

        max_new_tokens = max_new_tokens or self.config['model']['max_new_tokens']

        # Create inference prompt
        prompt = f"""### Legal Document Summarization

**Instruction:** Summarize the following legal judgment concisely, highlighting the key facts, legal issues, reasoning, and conclusion.

**Judgment:**
{judgment}

**Summary:**"""

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config['model']['max_length'] - max_new_tokens
        )

        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.config['model']['temperature'],
                top_p=self.config['model']['top_p'],
                do_sample=self.config['model']['do_sample'],
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract summary (everything after "**Summary:**")
        if "**Summary:**" in generated_text:
            summary = generated_text.split("**Summary:**")[-1].strip()
        else:
            summary = generated_text

        return summary

    def save_model(self, save_path: str):
        """Save the fine-tuned model."""
        if self.peft_model is None:
            raise ValueError("No fine-tuned model to save.")

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        self.peft_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        logger.info(f"Model saved to {save_path}")

    def load_fine_tuned_model(self, model_path: str):
        """Load a fine-tuned model."""
        from peft import PeftModel

        # Load base model first
        if self.model is None:
            self.load_base_model()

        # Load LoRA weights
        self.peft_model = PeftModel.from_pretrained(self.model, model_path)

        logger.info(f"Fine-tuned model loaded from {model_path}")

    @classmethod
    def from_pretrained(cls, model_path: str, config_path: str = "config/model_config.yaml"):
        """Load a pre-trained legal summarizer."""
        summarizer = cls(config_path)
        summarizer.load_base_model()
        summarizer.load_fine_tuned_model(model_path)
        return summarizer

def main():
    """Example usage of the legal summarizer."""
    # Initialize summarizer
    summarizer = LegalSummarizer()

    # Load base model
    summarizer.load_base_model()

    # Setup LoRA
    summarizer.setup_lora()

    # Example judgment text
    sample_judgment = """
    Appeal No. LXVI of 1949.
    This is an appeal against a judgment of the High Court of Judicature at Bombay in an income tax matter
    and it raises the question whether municipal property tax and urban immoveable property tax payable
    under the relevant Bombay Acts are allowable deductions under section 9 (1) (iv) of the Indian Income tax Act.
    """

    # Generate summary (before fine-tuning)
    summary = summarizer.summarize(sample_judgment)
    print("Generated Summary:", summary)

if __name__ == "__main__":
    main()
