from collections import deque
from itertools import islice
# import tiktoken
from transformers import AutoTokenizer
import torch
from datasets import load_dataset
from config import MAYFEI_SMALL


class StreamingPipeline:
    def __init__(self, context_length: int, batch_size: int, shuffle_buffer: int = 1000, seed: int = 42):
        self.context_length = context_length
        self.batch_size = batch_size
        # self.tokenizer = tiktoken.get_encoding('gpt2')
        self.tokenizer = AutoTokenizer.from_pretrained(
            MAYFEI_SMALL['tokenizer_name'], trust_remote_code=True, use_fast=False,
        )
        # self.eos_token_id = self.tokenizer.eot_token
        self.eos_token_id = self.tokenizer.eos_token_id

        dataset = load_dataset('openbmb/Ultra-FineWeb', streaming=True)
        self.en_dataset = dataset['en'].select_columns(['content'])
        self.zh_dataset = dataset['zh'].select_columns(['content'])

        if shuffle_buffer > 0:
            self.en_dataset = self.en_dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)
            self.zh_dataset = self.zh_dataset.shuffle(buffer_size=shuffle_buffer, seed=seed+1)

        self.en_iterator = iter(self.en_dataset)
        self.zh_iterator = iter(self.zh_dataset)

        self.en_tokens = deque()
        self.zh_tokens = deque()

        # positions: en, en, en, zh, zh
        self.language_index = 0

    def _next_text(self, language: str) -> str:
        iterator = self.en_iterator if language == 'en' else self.zh_iterator

        while True:
            sample = next(iterator)
            text = sample.get('content')
            if isinstance(text, str) and text.strip():
                return text.strip()

    def _next_sequence(self, language: str):
        token_buffer = self.en_tokens if language == 'en' else self.zh_tokens
        required_tokens = self.context_length + 1
        while len(token_buffer) < required_tokens:
            text = self._next_text(language)
            # token_buffer.extend(self.tokenizer.encode_ordinary(text))
            token_buffer.extend(self.tokenizer.encode(text, add_special_tokens=False))
            token_buffer.append(self.eos_token_id)
        block = list(islice(token_buffer, required_tokens))
        for _ in range(self.context_length):
            token_buffer.popleft()
        return torch.tensor(block[:-1], dtype=torch.long), torch.tensor(block[1:], dtype=torch.long)

    def next_batch(self):
        inputs = []
        targets = []

        for _ in range(self.batch_size):
            language = 'en' if self.language_index < 3 else 'zh'
            self.language_index = ( self.language_index + 1 ) % 5
            input_ids, target_ids = self._next_sequence(language)
            inputs.append(input_ids)
            targets.append(target_ids)
        return torch.stack(inputs), torch.stack(targets)

    def _state_dict(self):
        return {
            'en_dataset': self.en_dataset.state_dict(),
            'zh_dataset': self.zh_dataset.state_dict(),
            'en_tokens': list(self.en_tokens),
            'zh_tokens': list(self.zh_tokens),
            'language_index': self.language_index
        }

    def load_state_dict(self, state):
        self.en_dataset.load_state_dict(state['en_dataset'])
        self.zh_dataset.load_state_dict(state['zh_dataset'])
        self.en_iterator = iter(self.en_dataset)
        self.zh_iterator = iter(self.zh_dataset)

        self.en_tokens = deque(state['en_tokens'])
        self.zh_tokens = deque(state['zh_tokens'])

        self.language_index = state['language_index']

if __name__ == '__main__':
    # pipeline = StreamingPipeline(context_length=256, batch_size=10, shuffle_buffer=0)
    # i, t = pipeline.next_batch()
    # print(i.shape)
    # print(t.shape)
    tokenizer = AutoTokenizer.from_pretrained(
        "Skywork/Skywork-13B-base", trust_remote_code=True, use_fast=False
    )

    print(len(tokenizer))
    print(tokenizer.eos_token_id)
    a = tokenizer.encode("MayFei is bilingual.", add_special_tokens=False)
    b = tokenizer.encode("MayFei 是一个双语模型。", add_special_tokens=False)
    print(a)
    print(b)

    print(tokenizer.decode(a, skip_special_tokens=False))
    print(tokenizer.decode(b, skip_special_tokens=False))

