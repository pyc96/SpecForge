import re
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch
from transformers import PreTrainedTokenizer

from .template import ChatTemplate

__all__ = ["GeneralParser", "HarmonyParser"]


class Parser(ABC):

    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate):
        self.tokenizer = tokenizer
        self.chat_template = chat_template

    @abstractmethod
    def parse(
        self, conversation: "Conversation", max_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse the conversation into a list of tensors.

        Args:
            conversation: The conversation to parse.

        Returns:
            A list of tensors: [input_ids, loss_mask]
        """
        pass


_harmony_encoding = None


class GeneralParser(Parser):

    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate, native_parsing: bool):
        super().__init__(tokenizer, chat_template)
        self.system_prompt = chat_template.system_prompt
        self.user_message_separator = (
            f"{chat_template.end_of_turn_token}{chat_template.user_header}"
        )
        self.assistant_message_separator = (
            f"{chat_template.end_of_turn_token}{chat_template.assistant_header}"
        )
        self.native_parsing = native_parsing
        self.value_map = {"assistant": 1}

    def native_parse(self, messages, max_len):
        tokenizer = self.tokenizer
        conversation = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        if messages[0]["role"] == "system":
            si = messages[0]["content"]
            si_count = len(tokenizer(si, add_special_tokens=False).input_ids)
            si_header_count = 3  #  <|im_system|>system<|im_middle|>
            l = si_count + si_header_count
            msg_idx = [("system", si_header_count, l + 1)]
        else:
            msg_idx = []
            l = 0

        for msg in messages:
            if msg["role"] == "system":
                continue
            l += 4  # <|im_end|><|im_user|>user<|im_middle|>
            start = l
            l += len(tokenizer(msg["content"], add_special_tokens=False).input_ids)
            if msg["role"] == "assistant":
                l += 1 # for <think>
            msg_idx.append((msg["role"], start + 1 if msg["role"] == "assistant" else start, l + 2)) # to also include eos
            if l > max_len:
                break

        input_ids = tokenizer(
            conversation,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]
        loss_mask = torch.zeros_like(input_ids)

        for i in range(len(msg_idx)):
            role, start, end = msg_idx[i]
            if role not in self.value_map:
                continue
            loss_mask[start:end] = self.value_map[role]
        return input_ids[None, :max_len], loss_mask[None, :max_len], conversation

    def parse(
        self,
        conversation: "Conversation",
        max_length: int,
        preformatted: bool = False,
        **kwargs,
    ) -> Dict[str, List[torch.Tensor]]:
        if self.native_parsing:
            return self.native_parse(conversation, max_length)

        if not preformatted:
            messages = []

            if conversation[0]["role"] == "system":
                warnings.warn(
                    f"The first message is from system, we will use the system prompt from the data and ignore the system prompt from the template"
                )
                messages.append(
                    {"role": "system", "content": conversation[0]["content"]}
                )
                conversation = conversation[1:]
            else:
                if self.system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})

            convroles = ["user", "assistant"]
            for j, sentence in enumerate(conversation):
                role = sentence["role"]
                if role != convroles[j % 2]:
                    warnings.warn(
                        f"Conversation truncated due to unexpected role '{role}'. Expected '{convroles[j % 2]}'."
                    )
                    break
                messages.append(sentence)

            conversation = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, **kwargs
            )

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        encoding = self.tokenizer(
            conversation,
            return_offsets_mapping=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids[0]
        offsets = encoding.offset_mapping[0]
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

        # Find spans of assistant responses using regex
        assistant_pattern = (
            re.escape(self.assistant_message_separator)
            + r"(.*?)(?="
            + re.escape(self.user_message_separator)
            + "|$)"
        )
        for match in re.finditer(assistant_pattern, conversation, re.DOTALL):
            # Assistant response text span (excluding assistant_header itself)
            assistant_start_char = match.start(1)
            assistant_end_char = match.end(1)

            # Mark tokens overlapping with assistant response
            for idx, (token_start, token_end) in enumerate(offsets):
                # Token is part of the assistant response span
                if token_end <= assistant_start_char:
                    continue  # token before assistant text
                if token_start > assistant_end_char:
                    continue  # token after assistant text
                loss_mask[idx] = 1
        return input_ids, loss_mask, conversation


class HarmonyParser(Parser):

    def build_single_turn_prompt(
        self,
        user_msg: str,
        analysis_message: str,
        commentary_message: str,
        final_message: str,
        reasoning_level: str,
    ) -> str:
        """Embed user message into the required prompt template."""

        prompt_text = f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-06-28\n\nReasoning: {reasoning_level.lower()}\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
        prompt_text += f"<|start|>user<|message|>{user_msg}<|end|>"
        if analysis_message:
            prompt_text += f"<|start|>assistant<|channel|>analysis<|message|>{analysis_message}<|end|>"
        if commentary_message:
            prompt_text += f"<|start|>assistant<|channel|>commentary<|message|>{commentary_message}<|end|>"
        if final_message:
            prompt_text += (
                f"<|start|>assistant<|channel|>final<|message|>{final_message}<|end|>"
            )
        return prompt_text

    def parse(
        self, conversation: "Conversation", max_length: int, preformatted: bool = False
    ) -> List[torch.Tensor]:
        if not preformatted:
            user_message = None
            analysis_message = None
            commentary_message = None
            final_message = None
            reasoning_level = "Low"

            for j, message in enumerate(conversation):
                if message["role"] == "user":
                    user_message = message["content"]
                if message["role"] == "assistant_analysis":
                    analysis_message = message["content"]
                elif message["role"] == "assistant_commentary":
                    commentary_message = message["content"]
                elif message["role"] == "assistant_final":
                    final_message = message["content"]
                elif message["role"] == "assistant_reasoning_effort":
                    reasoning_level = message["content"]

            conversation = self.build_single_turn_prompt(
                user_message,
                analysis_message,
                commentary_message,
                final_message,
                reasoning_level,
            )

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        encoding = self.tokenizer(
            conversation,
            return_offsets_mapping=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids[0]
        offsets = encoding.offset_mapping[0]
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

        # Find spans of assistant responses using regex
        response = "<|end|>".join(
            conversation.split("<|end|><|start|>user<|message|>")[1].split("<|end|>")[
                1:
            ]
        )
        num_response_chars = len(response)
        num_system_chars = len(conversation) - num_response_chars

        # Mark tokens overlapping with assistant response
        for idx, (char_start, char_end) in enumerate(offsets):
            if char_end <= num_system_chars:
                continue
            loss_mask[idx] = 1
        return input_ids, loss_mask
