"""Includes utils such as Tokenizer and Dataset class for museC."""

import copy
import random
import torch
import pyroll

import torch.nn.functional as F

from .model import ModelConfig


class Tokenizer:
    def __init__(self, config: ModelConfig, pitch_aug_range: int):
        self.config = config
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
        config.vocab_size = self.vocab_size

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

        max_seq_len = self.config.max_seq_len
        stride_len = self.config.stride_len
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
        max_seq_len = self.config.max_seq_len

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
                # random.shuffle(buffer)  # UNCOMMENT

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

        # Format into tensors
        src_enc = self.encode(src)
        tgt_enc = torch.zeros(len(tgt), self.vocab_size)
        for i, labels in enumerate(tgt):
            num_labels = len(labels)
            for label_idx in self.encode(labels):
                tgt_enc[i, label_idx] = 1.0 / num_labels

        return src_enc, tgt_enc


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        pass


def main():
    import mido

    mid = mido.MidiFile("chopin.mid")
    piano_roll = pyroll.PianoRoll.from_midi(mid, 4)

    config = Config()
    tokenizer = Tokenizer(config, pitch_aug_range=0)

    seqs = tokenizer.format(piano_roll)
    seq = seqs[0]
    print(seq)

    src, tgt = tokenizer.apply(seq)


if __name__ == "__main__":
    main()
