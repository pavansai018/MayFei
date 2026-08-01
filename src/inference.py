from __future__ import annotations
import argparse
import gc
from pathlib import Path
# import tiktoken
from transformers import AutoTokenizer
import torch
from config import MAYFEI_INFERENCE_CONFIG, MAYFEI_SMALL
from gpt_model import GPTModel
from typing import Any


class MayFeiInference:
    def __init__(self, device: str | None = None):
        self.checkpoint_path = Path(MAYFEI_INFERENCE_CONFIG['default_checkpoint'])
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {self.checkpoint_path}')

        self.device = self._resolve_device(device)
        # self.tokenizer = tiktoken.get_encoding('gpt2')
        self.tokenizer = AutoTokenizer.from_pretrained(
            MAYFEI_SMALL['tokenizer_name'], trust_remote_code=True, use_fast=False,
        )
        # self.tokenizer_vocab_size = self.tokenizer.n_vocab
        self.tokenizer_vocab_size = len(self.tokenizer)
        # self.eos_token_id = self.tokenizer.eot_token
        self.eos_token_id = self.tokenizer.eos_token_id

        self.model_config = MAYFEI_SMALL.copy()

        self.model = self._load_model()
        print(f'Inference Ready | device={self.device}| checkpoint={self.checkpoint_path}| context_length={MAYFEI_INFERENCE_CONFIG['context_length']:,}')

    @staticmethod
    def _resolve_device(requested_device: str | None) -> torch.device:
        if requested_device is None:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if requested_device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('cuda was requested but cuda is not available')
        return torch.device(requested_device)
    

    def _load_checkpoint(self) -> Any:
        '''
        Load the checkpoint in CPU first.
        map=True reduces unnecessary copying when supporrted by the installed PyTorch version.
        '''
        try:
            return torch.load(self.checkpoint_path, map_location='cpu', weights_only=False, map=True)
        except (TypeError, RuntimeError):
            return torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
        '''
        supported formats:
            1. Lightning checkpoint:
                {
                    "state_dict": {
                        "model.token_embeddings.weight": ...
                    }
                }

            2. Custom training checkpoint:
                {
                    "model_state_dict": {...}
                }

            3. Direct model state dictionary:
                torch.save(model.state_dict(), path)
        '''
        if not isinstance(checkpoint, dict):
            raise TypeError('checkpoint must contain a dictionary')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_doct']
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise TypeError('extracted model state is not a dictionary')

        return state_dict

    @staticmethod
    def _remove_training_prefix(state_dict: dict[str, torch.Tensor],) -> dict[str, torch.Tensor]:
        '''
        MayFeiTrainer stores the GPT model as self.model.

        Lightning therefore saves keys such as:

            model.token_embeddings.weight

        GPTModel expects:

            token_embeddings.weight
        
        '''
        cleaned_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            cleaned_key = key
            if cleaned_key.startswith('model.'):
                cleaned_key = cleaned_key[len('model.'):]
            if cleaned_key.startswith('_orig_mod.'):
                cleaned_key = cleaned_key[len('_orig_mod.'):]

            cleaned_state_dict[cleaned_key] = value

        return cleaned_state_dict


    def _load_model(self) -> GPTModel:
        checkpoint = self._load_checkpoint()
        raw_state_dict = self._extract_state_dict(checkpoint)
        model_state_dict = self._remove_training_prefix(raw_state_dict)
        model = GPTModel(self.model_config)
        incompatible_keys = model.load_state_dict(model_state_dict, strict=True)
        if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
            raise RuntimeError(f'checkpoint does not exactly match with GPTModel.\nMissing Keys: {incompatible_keys.missing_keys}\nUnexpected Keys: {incompatible_keys.unexpected_keys}')

        del checkpoint
        del raw_state_dict
        del model_state_dict
        gc.collect()
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _apply_repetition_penalty(logits: torch.Tensor, generated_ids: torch.Tensor, repetition_penalty: float) -> torch.Tensor:
        if repetition_penalty == 1.0:
            return logits
        previous_token_ids = torch.unique(generated_ids)
        previous_logits = logits[:, previous_token_ids]
        penalized_logits = torch.where(
            previous_logits < 0,
            previous_logits * repetition_penalty,
            previous_logits / repetition_penalty,
        )

        logits[:, previous_token_ids] = penalized_logits
        return logits

    @staticmethod
    def _apply_top_k(logits: torch.Tensor, top_k: int | None,) -> torch.Tensor:
        if top_k is None or top_k <= 0:
            return logits
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k=top_k, dim=-1).values[:, -1].unsqueeze(-1)
        return torch.where(logits<threshold, torch.full_like(logits, float('-inf')), logits)

    @staticmethod
    def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        if top_p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
        remove_mask = cumulative_probabilities > top_p
        remove_mask[:, 1:] = remove_mask[:,:-1].clone()
        remove_mask[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove_mask, float('-inf'))
        filtered_logits = torch.full_like(logits, float('-inf'))
        filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        return filtered_logits

    def _sample_next_token(self, logits: torch.Tensor, generated_ids: torch.Tensor, temperature: float, top_k: int|None, top_p:float, repetition_penalty: float) -> torch.Tensor:
        logits = logits[:, :self.tokenizer_vocab_size].clone()
        logits = self._apply_repetition_penalty(logits=logits, generated_ids=generated_ids, repetition_penalty=repetition_penalty)

        # temp 0 means deterministic greedy decoding
        if temperature == 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / temperature
        logits = self._apply_top_k(logits=logits, top_k=top_k)
        logits = self._apply_top_p(logits=logits, top_p=top_p)
        probabilities = torch.softmax(logits, dim=-1)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError('generation probabilities contain NaN or Inf')
        return torch.multinomial(probabilities, num_samples=1)

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float=0.8, top_k: int|None=50, top_p:float=0.95, repetition_penalty:float=1.0, seed: int|None=None) -> str:
        if not prompt.strip():
            raise ValueError('prompt cannot be empty')
        if max_new_tokens <= 0:
            raise ValueError('max new tokens must be > 0')

        if temperature < 0:
            raise ValueError('temperature must be >= 0')

        if not 0.0 < top_p <= 1.0:
            raise ValueError('top_p must be in range (0, 1]')

        if repetition_penalty < 1.0:
            raise ValueError('repeteition penalty must be atleast 1.0')

        # if seed is not None:
        #     torch.manual_seed(seed)

        #     if torch.cuda.is_available():
        #         torch.cuda.manual_seed_all(seed)

        prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        # leave at least one position for generation
        maximum_prompt_length = MAYFEI_INFERENCE_CONFIG['context_length'] - 1
        if len(prompt_token_ids) > maximum_prompt_length:
            prompt_token_ids = prompt_token_ids[-maximum_prompt_length:]

            print('Prompt exceeded the context window and was truncated from the left')
        input_ids = torch.tensor(prompt_token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        generated_token_ids: list[int] = []

        for _ in range(max_new_tokens):
            model_input = input_ids[:, -MAYFEI_INFERENCE_CONFIG['context_length']:]
            logits = self.model(model_input)
            if isinstance(logits, tuple):
                logits = logits[0]
            if hasattr(logits, 'logits'):
                logits = logits.logits

            if logits.ndim != 3:
                raise RuntimeError('GPT model must return logits with shape [batch, sequence, vocab_size]')

            next_token_logits = logits[:, -1, :,]
            next_token_id = self._sample_next_token(
                logits=next_token_logits,
                generated_ids=input_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            token_id = int(next_token_id.item())
            if token_id == self.eos_token_id:
                break
            generated_token_ids.append(token_id)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        return self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)



def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text using a trained MayFei checkpoint.")
    parser.add_argument("--checkpoint", default=MAYFEI_INFERENCE_CONFIG['default_checkpoint'])
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    torch.manual_seed(args.seed)

    inference = MayFeiInference(args.device)
    top_k = args.top_k if args.top_k > 0 else None

    if args.prompt:
        output = inference.generate(
            args.prompt, args.max_new_tokens, args.temperature,
            top_k, args.top_p, args.repetition_penalty,
        )
        print(f"\nPrompt:\n{args.prompt}\n\nGenerated continuation:\n{output}")
        return

    print("\nInteractive inference — type 'quit' to stop.\n")

    while True:
        prompt = input("Prompt: ").strip()

        if prompt.lower() in {"quit", "exit", "q"}:
            break

        if prompt:
            output = inference.generate(
                prompt, args.max_new_tokens, args.temperature,
                top_k, args.top_p, args.repetition_penalty, args.seed
            )
            print(f"\nGenerated continuation:\n{output}\n")


if __name__ == "__main__":
    main()
