"""Includes utils such as Tokenizer and Dataset class for museC."""

import copy
import json
import random
import torch
import pyroll

import torch.nn.functional as F

from model import ModelConfig


class Tokenizer:
    def __init__(self, model_config: ModelConfig, pitch_aug_range: int = 4):
        self.model_config = model_config
        self.pitch_aug_range = pitch_aug_range

        # Special tokens
        self.eos_tok = "<E>"
        self.bos_tok = "<S>"
        self.pad_tok = "<P>"
        self.unk_tok = "<U>"
        self.time_tok = "<T>"

        special_toks = [
            self.eos_tok,
            self.bos_tok,
            self.pad_tok,
            self.unk_tok,
            self.time_tok,
        ]

        # We use tuples because they are hashable (pyroll uses dicts)
        legato_notes = [(i, "l") for i in range(128)]
        staccato_notes = [(i, "s") for i in range(128)]

        self.vocab = legato_notes + staccato_notes + special_toks
        self.tok_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_tok = {i: tok for i, tok in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

        # Adjust model_config settings
        model_config.vocab_size = self.vocab_size
        model_config.pad_id = self.tok_to_id[self.pad_tok]

    def encode(self, src: list):
        """Encodes a formatted piano-roll."""

        def enc_fn(tok):
            return self.tok_to_id.get(tok, self.tok_to_id[self.unk_tok])

        return torch.tensor([enc_fn(tok) for tok in src])

    def decode(self, src: torch.Tensor):
        """Decodes a (encoded) formatted piano-roll."""

        def dec_fn(tok):
            return self.id_to_tok.get(tok, self.tok_to_id[self.unk_tok])

        if isinstance(src, torch.Tensor):
            dec = [dec_fn(idx) for idx in src.tolist()]
        else:
            dec = [dec_fn(idx) for idx in src]

        return dec

    def format(self, piano_roll: pyroll.PianoRoll):
        """Formats piano-roll into sequential form."""

        def _get_seq(roll: list, idx: int, beginning: bool = False):
            """Returns formatted (sequential) 'slice' of piano-roll."""
            # Add bos token if beginning is True
            if beginning is True:
                seq = [self.bos_tok]
            else:
                seq = []

            # -1 counts for a possible time_tok or eos_tok
            while idx < len(roll) and (
                len(seq) + len(roll[idx]) <= max_seq_len - 1
            ):
                for note in roll[idx]:
                    seq.append((note["val"], note["art"]))  # Convert to tuple

                seq.append(self.time_tok)
                idx += 1

            # If end of piano-roll, append end of sequence
            if idx == len(roll):
                seq.pop()  # Remove last time_tok
                seq.append(self.eos_tok)

            # Pad to max_seq_len
            seq = seq + [self.pad_tok] * (max_seq_len - len(seq))

            return seq

        max_seq_len = self.model_config.max_seq_len
        stride_len = self.model_config.stride_len
        roll = copy.deepcopy(piano_roll.roll)

        # Calculates cumulative sum of chord lengths
        chord_len = [len(chord) for chord in roll]
        cum_sum = [
            sum(chord_len[: i + 1]) + (i + 1) for i in range(len(chord_len))
        ]

        # Calculates chord idxs to start sequences from.
        curr, prev = 0, 0
        idxs = []
        while True:
            while curr < len(cum_sum) and (
                cum_sum[curr] - cum_sum[prev] <= stride_len
            ):
                curr += 1

            idxs.append(prev)
            if cum_sum[-1] - cum_sum[prev] <= max_seq_len:
                break
            else:
                prev = curr

        sequences = [_get_seq(roll, idxs[0], beginning=True)]
        for idx in idxs[1:]:
            sequences.append(_get_seq(roll, idx))

        return sequences

    def apply(self, seq: list):
        """Returns (src, tgt) pair for training."""

        def _aug_buffer(buffer: list):
            """Applies pitch augmentation to buffer (chord)."""

            return [(entry[0] + pitch_aug, entry[1]) for entry in buffer]

        pitch_aug = random.randint(-self.pitch_aug_range, self.pitch_aug_range)
        max_seq_len = self.model_config.max_seq_len

        src = []
        tgt = []

        idx = 0
        buffer_idx = 1
        while buffer_idx < max_seq_len:
            # Load chord into buffer
            buffer = []
            while (
                seq[buffer_idx] != self.time_tok  # noqa
                and seq[buffer_idx] != self.eos_tok  # noqa
                and seq[buffer_idx] != self.bos_tok  # noqa
                and seq[buffer_idx] != self.pad_tok  # noqa
            ):
                buffer.append(seq[buffer_idx])
                buffer_idx += 1

            # Add shuffled chord to src, tgt
            if not buffer:
                src.append(seq[idx])
                tgt.append([seq[idx + 1]])
            else:
                buffer = _aug_buffer(buffer)
                # random.shuffle(buffer)  # DEBUG

                # Add time_tok (or bos_tok or eos_tok) to src, tgt
                src.append(seq[idx])
                tgt.append(buffer)

                # Add buffer note to src, tgt
                for i, tok in enumerate(buffer):
                    src.append(tok)
                    tgt.append(buffer[i + 1 :].copy())

                # Add time_tok target to last entry in buffer
                tgt[-1] = [self.time_tok]

            idx = buffer_idx
            buffer_idx = idx + 1

        # Add last tokens to src, tgt
        src.append(seq[idx])
        tgt.append([self.pad_tok])

        # Format into tensors (not very efficient)
        src_enc = self.encode(src)
        tgt_enc = torch.zeros(len(tgt), self.vocab_size)
        for i, labels in enumerate([self.encode(labels) for labels in tgt]):
            tgt_enc[i, labels] = 1 / len(labels)

        return src_enc, tgt_enc


class Dataset(torch.utils.data.Dataset):
    """PyTorch Dataset class for training data.

    Args:
        dataset (Dataset): Dataset (PianoRoll) for training.
        tokenizer (model.Tokenizer): Tokenizer subclass holding methods for
            PianoRoll to torch.Tensor conversion.
        split (str): Whether to use train or test set.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        data: list = [],
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.tokenizer.apply(self.data[idx])

    @classmethod
    def from_pianoroll_dataset(
        cls,
        dataset: pyroll.PianoRollDataset,
        tokenizer: Tokenizer,
    ):
        data = []
        for piano_roll in dataset.data:
            data += tokenizer.format(piano_roll)

        return Dataset(tokenizer, data)

    @classmethod
    def from_json(
        cls,
        load_path: str,
        tokenizer: Tokenizer,
        key: str | None = None,
    ):
        with open(load_path) as f:
            if key is None:
                data = json.load(f)
            else:
                data = json.load(f)[key]

        # Converts lists back into tuples (json converts tuples to lists)
        for seq in data:
            for i, tok in enumerate(seq):
                if isinstance(tok, list):
                    seq[i] = tuple(tok)

        assert isinstance(data, list), "Loaded data must be a list."
        assert (
            len(data[0]) == tokenizer.model_config.max_seq_len
        ), "Sequence len mismatch."

        return Dataset(tokenizer, data)
