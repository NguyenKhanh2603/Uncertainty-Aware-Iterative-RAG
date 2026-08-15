"""LLM/VLM API client with logprobs support, batched generation, and cost tracking."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from uncertainty_rag.utils.cost_tracker import CostTracker

class ChunkStatus(str, Enum):
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    FULLY_TRUNCATED = "FULLY_TRUNCATED"

@dataclass
class ChunkSpan:
    chunk_idx: int
    chunk_id: str
    modality: str
    start: Optional[int]
    end: Optional[int]
    status: ChunkStatus

@dataclass
class AlignmentResult:
    inputs: Any
    batch_chunk_spans: list[list[ChunkSpan]]
    prompt_strings: list[str]

class ChunkAlignmentError(Exception):
    pass


# ── Data Structures ─────────────────────────────────────────────────────────────


@dataclass
class TokenLogprob:
    """A single token with its log-probability."""

    token: str
    logprob: float
    top_logprobs: Optional[list[dict[str, float]]] = None


@dataclass
class SampleResult:
    """Result of a single LLM generation."""

    text: str
    token_logprobs: list[TokenLogprob] = field(default_factory=list)
    finish_reason: str = "stop"


# ── Abstract Base ───────────────────────────────────────────────────────────────


class BaseLLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        n: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        logprobs: bool = True,
        top_logprobs: int = 5,
        json_mode: bool = False,
    ) -> list[SampleResult]:
        """Generate n completions for the given messages."""
        ...


# ── OpenAI Implementation ───────────────────────────────────────────────────────


class OpenAIClient(BaseLLMClient):
    """OpenAI-compatible API client (works with OpenAI, Azure, vLLM, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cost_tracker: Optional[CostTracker] = None,
    ) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.cost_tracker = cost_tracker or CostTracker()

    def _has_image_content(self, messages: list[dict]) -> bool:
        """Check if messages contain image_url content blocks."""
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return True
        return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
    def _call_api(
        self,
        messages: list[dict],
        n: int,
        temperature: float,
        max_tokens: int,
        logprobs: bool,
        top_logprobs: int,
        json_mode: bool,
    ) -> Any:
        """Single API call with retry logic."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = top_logprobs
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        return self.client.chat.completions.create(**kwargs)

    def generate(
        self,
        messages: list[dict[str, Any]],
        n: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        logprobs: bool = True,
        top_logprobs: int = 5,
        json_mode: bool = False,
    ) -> list[SampleResult]:
        """Generate n completions.

        For multimodal messages (containing images), n>1 may not be supported.
        In that case, fall back to n sequential calls.
        """
        has_images = self._has_image_content(messages)

        if has_images and n > 1:
            # VLM limitation: n>1 often unsupported with images → sequential calls
            return self._generate_sequential(
                messages, n, temperature, max_tokens, logprobs, top_logprobs, json_mode
            )

        t0 = time.time()
        response = self._call_api(
            messages, n, temperature, max_tokens, logprobs, top_logprobs, json_mode
        )
        latency = time.time() - t0

        results = []
        for choice in response.choices:
            token_logprobs_list = []
            if logprobs and choice.logprobs and choice.logprobs.content:
                for token_info in choice.logprobs.content:
                    top_lps = None
                    if token_info.top_logprobs:
                        top_lps = [
                            {"token": t.token, "logprob": t.logprob}
                            for t in token_info.top_logprobs
                        ]
                    token_logprobs_list.append(
                        TokenLogprob(
                            token=token_info.token,
                            logprob=token_info.logprob,
                            top_logprobs=top_lps,
                        )
                    )

            results.append(
                SampleResult(
                    text=choice.message.content or "",
                    token_logprobs=token_logprobs_list,
                    finish_reason=choice.finish_reason or "stop",
                )
            )

        # Track cost
        usage = response.usage
        if usage:
            self.cost_tracker.record_call(
                model=self.model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                num_completions=n,
                latency_s=latency,
            )

        return results

    def _generate_sequential(
        self,
        messages: list[dict],
        n: int,
        temperature: float,
        max_tokens: int,
        logprobs: bool,
        top_logprobs: int,
        json_mode: bool,
    ) -> list[SampleResult]:
        """Fallback: generate n samples via n sequential API calls."""
        all_results = []
        for _ in range(n):
            results = self.generate(
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                json_mode=json_mode,
            )
            all_results.extend(results)
        return all_results


# ── HuggingFace Implementation ──────────────────────────────────────────────────


class HuggingFaceLocalClient(BaseLLMClient):
    """Client for local HuggingFace models (e.g. Llama-3-8B).
    
    Required for advanced System-Level pruning strategies:
    - Attention Masking (Strategy 3)
    - Attention Saliency (Strategy 5)
    """

    def __init__(self, model_name: str, device: str = "cuda", load_in_4bit: bool = True):
        if not HF_AVAILABLE:
            raise ImportError("Please install torch and transformers to use HuggingFaceLocalClient.")
        
        self.model_name = model_name
        self.device = device
        
        model_kwargs = {"device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            
        # [CRITICAL FIX]: Force eager attention to allow attention weight extraction (for Saliency Pruning)
        model_kwargs["attn_implementation"] = "eager"
        
        # Check if Qwen2-VL
        if "qwen2-vl" in model_name.lower():
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                **model_kwargs
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.tokenizer = self.processor.tokenizer
            self.is_qwen_vl = True
            try:
                from qwen_vl_utils import process_vision_info
                self.process_vision_info = process_vision_info
            except ImportError:
                raise ImportError("Please install qwen-vl-utils for Qwen2-VL support.")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.processor = None
            self.is_qwen_vl = False
            
        self.model.eval()

    def _prepare_inputs(self, messages: list[dict[str, Any]]) -> tuple[Any, str]:
        if self.is_qwen_vl:
            qwen_messages = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    new_content = []
                    for part in content:
                        if part.get("type") == "text":
                            new_content.append({"type": "text", "text": part["text"]})
                        elif part.get("type") == "image_url":
                            # Translate OpenAI format to Qwen2-VL format
                            url = part["image_url"]["url"]
                            if url.startswith("file://"):
                                url = url.replace("file://", "")
                            # Giới hạn phân giải để cứu VRAM trên T4
                            new_content.append({"type": "image", "image": url, "max_pixels": 256 * 256})
                    qwen_messages.append({"role": msg["role"], "content": new_content})
                else:
                    qwen_messages.append(msg)
            
            text = self.processor.apply_chat_template(qwen_messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = self.process_vision_info(qwen_messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)
            return inputs, text
        else:
            processed_messages = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
                    content = "\n".join(text_parts)
                processed_messages.append({"role": msg["role"], "content": content})

            prompt = self.tokenizer.apply_chat_template(processed_messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            return inputs, prompt

    def generate(
        self,
        messages: list[dict[str, Any]],
        n: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        logprobs: bool = True,
        top_logprobs: int = 5,
        json_mode: bool = False,
    ) -> list[SampleResult]:
        """Generate samples from local model."""
        inputs, _ = self._prepare_inputs(messages)
        
        # Generation config
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "num_return_sequences": 1,  # CHỈ SINH 1 MẪU MỖI LẦN ĐỂ TIẾT KIỆM RAM
            "output_scores": True if logprobs else False,
            "return_dict_in_generate": True,
        }
        
        # Ép LLM sinh đa dạng khi do_sample=True
        if temperature > 0:
            gen_kwargs["top_p"] = 0.85
            gen_kwargs["top_k"] = 50
            gen_kwargs["num_beams"] = 1
            gen_kwargs["repetition_penalty"] = 1.1
        
        results = []
        if n > 1:
            print(f"\n[DEBUG] Đang chạy Generate tuần tự M={n}, Temperature={temperature} (do_sample={temperature > 0})")
            
        with torch.no_grad():
            for i in range(n):
                outputs = self.model.generate(**inputs, **gen_kwargs)
                
                # Extract generated tokens (skip prompt)
                gen_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
                text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                
                if n > 1:
                    print(f"   [RAW SAMPLE {i+1}]: {text}")
                
                token_lps = []
                if logprobs:
                    # Calculate log probabilities
                    scores = outputs.scores
                    for j, token_id in enumerate(gen_tokens):
                        if token_id in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]: 
                            break
                        if j < len(scores):
                            log_probs = torch.nn.functional.log_softmax(scores[j][0], dim=-1)
                            token_prob = log_probs[token_id].item()
                            token_str = self.tokenizer.decode(token_id)
                            token_lps.append(TokenLogprob(token=token_str, logprob=token_prob))
                
                results.append(
                    SampleResult(
                        text=text,
                        token_logprobs=token_lps,
                        finish_reason="stop"
                    )
                )
                
        return results

    def align_and_prepare_inputs(self, query: str, chunks: list[Any]) -> AlignmentResult:
        """
        Token Alignment V2: Uses temporary markers and offset_mapping to perfectly
        align text/table chunks, and vision tokens for image chunks.
        """
        messages_content = [{"type": "text", "text": query + "\n\nContext:\n"}]
        markers_info = []
        
        for i, chunk in enumerate(chunks):
            modality = getattr(chunk, "modality", "text")
            chunk_id = getattr(chunk, "id", f"chunk_{i}")
            
            if modality == "image":
                if self.is_qwen_vl:
                    # Qwen2-VL expects url or base64
                    from pathlib import Path
                    image_path = chunk.content
                    if image_path.startswith("file://"):
                        image_path = image_path[7:]
                    if Path(image_path).exists() or image_path.startswith("http"):
                        messages_content.append({"type": "image", "image": chunk.content})
                    else:
                        messages_content.append({"type": "text", "text": f"[Image {i+1} Missing]"})
                markers_info.append({
                    "chunk_idx": i,
                    "chunk_id": chunk_id,
                    "modality": modality,
                })
            else:
                # Use Unicode Private Use Area (PUA) starting at U+E000
                start_marker = chr(0xE000 + i * 2)
                end_marker = chr(0xE000 + i * 2 + 1)
                
                if ord(end_marker) > 0xF8FF:
                    raise ChunkAlignmentError("Ran out of safe PUA markers.")
                    
                marked_text = f"{start_marker}{chunk.content}{end_marker}"
                messages_content.append({"type": "text", "text": marked_text + "\n"})
                markers_info.append({
                    "chunk_idx": i,
                    "chunk_id": chunk_id,
                    "modality": modality,
                    "start_marker": start_marker,
                    "end_marker": end_marker
                })
                
        messages = [{"role": "user", "content": messages_content}]
        
        # Apply chat template with markers
        prompt_with_markers = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Locate markers and remove them to create a clean prompt
        marker_positions = []
        for info in markers_info:
            if info["modality"] == "image":
                continue
            s_idx = prompt_with_markers.find(info["start_marker"])
            e_idx = prompt_with_markers.find(info["end_marker"])
            
            if s_idx == -1 or e_idx == -1:
                raise ChunkAlignmentError(f"Marker not found in rendered prompt for chunk {info['chunk_idx']}")
                
            marker_positions.append((s_idx, "start", info))
            marker_positions.append((e_idx, "end", info))
            
        marker_positions.sort(key=lambda x: x[0])
        
        clean_prompt = ""
        last_idx = 0
        char_spans = {} 
        
        for idx, m_type, info in marker_positions:
            clean_prompt += prompt_with_markers[last_idx:idx]
            if m_type == "start":
                char_spans[info["chunk_idx"]] = {"start": len(clean_prompt)}
            else:
                char_spans[info["chunk_idx"]]["end"] = len(clean_prompt)
            
            # Skip the marker character itself
            last_idx = idx + 1
            
        clean_prompt += prompt_with_markers[last_idx:]
        
        # Pass the clean prompt to the processor to get offset_mapping
        if self.is_qwen_vl:
            image_inputs, video_inputs = self.process_vision_info(messages)
            inputs = self.processor(
                text=[clean_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                return_offsets_mapping=True
            ).to(self.model.device)
        else:
            inputs = self.tokenizer(
                [clean_prompt],
                return_tensors="pt",
                return_offsets_mapping=True
            ).to(self.model.device)
            
        if "offset_mapping" not in inputs:
            raise ChunkAlignmentError("Processor did not return offset_mapping. Make sure you are using a Fast tokenizer.")
            
        offset_mapping = inputs["offset_mapping"][0].tolist()
        input_ids_list = inputs.input_ids[0].tolist()
        
        # Vision markers for Qwen2-VL
        vision_starts = []
        vision_ends = []
        if self.is_qwen_vl:
            try:
                vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
                vision_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
            except:
                vision_start_id = 151652
                vision_end_id = 151653
            vision_starts = [i for i, x in enumerate(input_ids_list) if x == vision_start_id]
            vision_ends = [i for i, x in enumerate(input_ids_list) if x == vision_end_id]
            
        chunk_spans = []
        image_occurrence = 0
        
        for info in markers_info:
            chunk_idx = info["chunk_idx"]
            chunk_id = info["chunk_id"]
            modality = info["modality"]
            
            if modality == "image":
                if not self.is_qwen_vl:
                    # Ignore image if model isn't VLM
                    chunk_spans.append(ChunkSpan(chunk_idx, chunk_id, modality, None, None, ChunkStatus.FULLY_TRUNCATED))
                elif image_occurrence < len(vision_starts) and image_occurrence < len(vision_ends):
                    token_start = vision_starts[image_occurrence]
                    token_end = vision_ends[image_occurrence] + 1
                    status = ChunkStatus.FULL
                    image_occurrence += 1
                    chunk_spans.append(ChunkSpan(chunk_idx, chunk_id, modality, token_start, token_end, status))
                else:
                    chunk_spans.append(ChunkSpan(chunk_idx, chunk_id, modality, None, None, ChunkStatus.FULLY_TRUNCATED))
                    
            else:
                char_start = char_spans[chunk_idx]["start"]
                char_end = char_spans[chunk_idx]["end"]
                
                token_start = None
                token_end = None
                
                for i, (tok_start, tok_end) in enumerate(offset_mapping):
                    if tok_start == tok_end:
                        continue
                    if tok_start < char_end and tok_end > char_start:
                        if token_start is None:
                            token_start = i
                        token_end = i + 1
                        
                if token_start is None:
                    status = ChunkStatus.FULLY_TRUNCATED
                else:
                    last_tok_end = offset_mapping[token_end - 1][1]
                    if last_tok_end < char_end:
                        status = ChunkStatus.PARTIAL
                    else:
                        status = ChunkStatus.FULL
                        
                chunk_spans.append(ChunkSpan(
                    chunk_idx=chunk_idx,
                    chunk_id=chunk_id,
                    modality=modality,
                    start=token_start,
                    end=token_end,
                    status=status
                ))
                
        # Remove offset_mapping from inputs because model.generate doesn't expect it
        del inputs["offset_mapping"]
        
        return AlignmentResult(
            inputs=inputs,
            batch_chunk_spans=[chunk_spans],
            prompt_strings=[clean_prompt]
        )

    def generate_with_custom_mask(
        self,
        query: str,
        chunks: list[Any],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        n: int = 3,
    ) -> list[list[SampleResult]]:
        """
        Strategy 3: Zero-Cost LOO via Attention Masking.
        Generates `n` samples for each masked chunk safely to avoid VRAM OOM.
        """
        alignment = self.align_and_prepare_inputs(query, chunks)
        inputs = alignment.inputs
        chunk_spans = alignment.batch_chunk_spans[0]
        
        K = len(chunk_spans)
        if K == 0:
            return []
            
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        
        all_results = []
        with torch.no_grad():
            for i, span in enumerate(chunk_spans):
                # Duplicate inputs `n` times for generating `n` samples for THIS specific chunk masking
                input_ids = inputs.input_ids.repeat(n, 1)
                if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
                    attention_mask = inputs.attention_mask.repeat(n, 1)
                else:
                    attention_mask = torch.ones_like(input_ids)
                
                # Copy images n times
                if "pixel_values" in inputs:
                    gen_kwargs["pixel_values"] = inputs.pixel_values.repeat(n, 1) if len(inputs.pixel_values.shape) > 1 else inputs.pixel_values
                    gen_kwargs["image_grid_thw"] = inputs.image_grid_thw.repeat(n, 1) if len(inputs.image_grid_thw.shape) > 1 else inputs.image_grid_thw
                
                # Apply masking for this chunk across all `n` rows
                if span.status != ChunkStatus.FULLY_TRUNCATED and span.start is not None and span.end is not None:
                    start_idx = max(0, span.start)
                    end_idx = min(attention_mask.shape[1], span.end)
                    attention_mask[:, start_idx:end_idx] = 0
                    
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )
                
                chunk_samples = []
                for j in range(n):
                    gen_tokens = outputs.sequences[j][input_ids.shape[1]:]
                    text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    chunk_samples.append(SampleResult(text=text, finish_reason="stop"))
                    
                all_results.append(chunk_samples)
                
        return all_results

    def extract_attention_saliency(
        self,
        query: str,
        chunks: list[Any],
        max_tokens: int = 50,
    ) -> list[float]:
        """
        Strategy 5: Attention Saliency.
        """
        alignment = self.align_and_prepare_inputs(query, chunks)
        inputs = alignment.inputs
        chunk_spans = alignment.batch_chunk_spans[0]
        
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "output_attentions": True,
            "return_dict_in_generate": True,
            "do_sample": False
        }
        if "pixel_values" in inputs:
            gen_kwargs["pixel_values"] = inputs.pixel_values
            gen_kwargs["image_grid_thw"] = inputs.image_grid_thw
            
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **gen_kwargs
            )
            
        if not outputs.attentions:
            raise ValueError("Model did not return attentions. Ensure output_attentions=True works for this model architecture.")
            
        scores = []
        for span in chunk_spans:
            if span.status == ChunkStatus.FULLY_TRUNCATED or span.start is None or span.end is None:
                scores.append(0.0)
                continue
                
            start_idx = max(0, span.start)
            end_idx = min(inputs.input_ids.shape[1], span.end)
            
            chunk_attention_sum = 0.0
            for token_attentions in outputs.attentions:
                if not token_attentions:
                    raise ValueError("Token attentions is empty. Model likely uses SDPA/FlashAttention which doesn't support attention weight extraction.")
                last_layer_attention = token_attentions[-1]
                
                # last_layer_attention shape: (1, num_heads, q_len, kv_len)
                # We want the average across heads for the newly generated token (last query)
                avg_heads = last_layer_attention[0].mean(dim=0) # shape: (q_len, kv_len)
                token_saliency = avg_heads[-1] # shape: (kv_len,)
                
                if start_idx < len(token_saliency):
                    actual_end = min(len(token_saliency), end_idx)
                    chunk_attention_sum += token_saliency[start_idx:actual_end].sum().item()
                    
            # Chuẩn hóa (Normalize) bằng cách chia cho số token được sinh ra
            # Để đảm bảo Saliency luôn nằm trong khoảng [0.0, 1.0] (tức là 0% đến 100%)
            if outputs.attentions:
                chunk_attention_sum /= len(outputs.attentions)
            
            scores.append(chunk_attention_sum)
            
        return scores
