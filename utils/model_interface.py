from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

import os
import json
import shutil
import subprocess
import urllib.request
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import openai
import httpx

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

load_dotenv()  # Load environment variables from .env if present


class BaseModel(ABC):
    @abstractmethod
    def generate_answer(self, question: str, **kwargs) -> str:
        pass


class GPT51Model(BaseModel):
    """OpenAI GPT-5.1 Chat Completions wrapper with reasoning support.

    Constraints (from design doc):
    - Default reasoning_effort is "medium"; supported values: none | low | medium | high.
    - When reasoning_effort != "none", do NOT send temperature/top_p.
    - Use max_completion_tokens for the cap (not max_tokens).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-5.1",
        default_reasoning_effort: str = "medium",
    ):
        self.client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model_name = model_name or "gpt-5.1"
        self.default_reasoning_effort = default_reasoning_effort or "medium"
        # Keep cap behavior aligned with Gemini defaults unless caller overrides.
        self.default_max_completion_tokens = int(os.environ.get("OPENAI_MAX_COMPLETION_TOKENS", "1024"))

    @staticmethod
    def _normalize_reasoning_effort(reasoning_effort: Optional[str], default: str = "medium") -> str:
        valid = {"none", "low", "medium", "high"}
        if reasoning_effort is None:
            return default
        lowered = str(reasoning_effort).lower()
        return lowered if lowered in valid else default

    @staticmethod
    def _build_messages(question: str, kwargs: Dict[str, Any]) -> List[Dict[str, str]]:
        if kwargs.get("messages"):
            return kwargs["messages"]

        system_prompt = kwargs.get("system")
        user_content = kwargs.get("user") or question

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _build_body(self, question: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        messages = self._build_messages(question, kwargs)
        reasoning_effort = self._normalize_reasoning_effort(
            kwargs.get("reasoning_effort"),
            default=self.default_reasoning_effort,
        )
        max_completion_tokens = (
            kwargs.get("max_completion_tokens")
            or kwargs.get("max_new_tokens")
            or kwargs.get("max_tokens")
            or self.default_max_completion_tokens
        )

        body: Dict[str, Any] = {
            "model": kwargs.get("model_name", self.model_name),
            "messages": messages,
            "reasoning_effort": reasoning_effort,
            "max_completion_tokens": max_completion_tokens,
        }

        stop = kwargs.get("stop")
        if stop is not None:
            body["stop"] = self._normalize_stop(stop)

        if reasoning_effort == "none":
            # Only forward sampling when explicitly allowed.
            if "temperature" in kwargs and kwargs["temperature"] is not None:
                body["temperature"] = kwargs["temperature"]
            if "top_p" in kwargs and kwargs["top_p"] is not None:
                body["top_p"] = kwargs["top_p"]


        return body

    @staticmethod
    def _normalize_stop(stop_val: Any) -> Any:
        """Ensure stop is JSON-serializable (str or List[str])."""
        if isinstance(stop_val, str):
            return stop_val
        if isinstance(stop_val, (list, tuple, set)):
            return [str(s) for s in stop_val]
        try:
            return [str(s) for s in list(stop_val)]
        except Exception:
            return str(stop_val)

    @staticmethod
    def _estimate_prompt_tokens(messages: List[Dict[str, str]]) -> int:
        joined = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
        return max(1, len(joined) // 4) if joined else 0

    def generate_answer(self, question: str, **kwargs) -> str:
        usage_meta = kwargs.get("usage_meta") or {}
        if "usage_meta" in kwargs:
            kwargs = dict(kwargs)
            kwargs.pop("usage_meta", None)
        body = self._build_body(question, kwargs)
        messages = body["messages"]
        reasoning_effort = body.get("reasoning_effort", self.default_reasoning_effort)

        try:
            response = self.client.chat.completions.create(**body)
        except openai.BadRequestError as exc:
            msg = str(exc)
            if any(token in msg for token in ("max_tokens", "output limit", "Could not finish the message")):
                _warn_empty_completion(
                    content="",
                    usage={},
                    model_name=body.get("model", self.model_name),
                    example_id=usage_meta.get("example_id") if isinstance(usage_meta, dict) else None,
                    regime=usage_meta.get("regime") if isinstance(usage_meta, dict) else None,
                    context="sync",
                )
                meta = {
                    "context": "generation",
                    "backend": "openai",
                    "reasoning_effort": reasoning_effort,
                    "temperature_forwarded": body.get("temperature") if reasoning_effort == "none" else None,
                    "top_p_forwarded": body.get("top_p") if reasoning_effort == "none" else None,
                    "error": "output_limit",
                    "error_message": msg,
                }
                if isinstance(usage_meta, dict):
                    meta.update(usage_meta)
                _log_usage(
                    model_name=body.get("model", self.model_name),
                    prompt_tokens=self._estimate_prompt_tokens(messages),
                    completion_tokens=0,
                    total_tokens=None,
                    meta=meta,
                )
                return ""
            raise

        text = response.choices[0].message.content if response.choices else ""
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None

        if not text:
            _warn_empty_completion(
                content=text,
                usage=usage,
                model_name=body.get("model", self.model_name),
                example_id=usage_meta.get("example_id") if isinstance(usage_meta, dict) else None,
                regime=usage_meta.get("regime") if isinstance(usage_meta, dict) else None,
                context="sync",
            )

        meta = {
            "context": "generation",
            "backend": "openai",
            "reasoning_effort": reasoning_effort,
            "temperature_forwarded": body.get("temperature") if reasoning_effort == "none" else None,
            "top_p_forwarded": body.get("top_p") if reasoning_effort == "none" else None,
        }
        if isinstance(usage_meta, dict):
            meta.update(usage_meta)

        _log_usage(
            model_name=body.get("model", self.model_name),
            prompt_tokens=prompt_tokens if prompt_tokens is not None else self._estimate_prompt_tokens(messages),
            completion_tokens=completion_tokens if completion_tokens is not None else (len(text) // 4 if text else 0),
            total_tokens=total_tokens,
            meta=meta,
        )
        return text or ""

    def create_batch_job(
        self,
        input_file_id: str,
        *,
        completion_window: str = "24h",
        metadata: Optional[Dict[str, str]] = None,
        endpoint: str = "/v1/chat/completions",
    ):
        """Create a Batch job targeting Chat Completions."""
        return self.client.batches.create(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=metadata,
        )


class LlamaModel(BaseModel):
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", load_in_8bit: bool = False):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_load_kwargs)

    def generate_answer(self, question: str, **kwargs) -> str:
        usage_meta = kwargs.get("usage_meta") or {}
        if "usage_meta" in kwargs:
            kwargs = dict(kwargs)
            kwargs.pop("usage_meta", None)
        temperature = kwargs.get("temperature", 0.0)
        top_p = kwargs.get("top_p", 0.95)
        top_k = kwargs.get("top_k", 64)
        formatted_prompt = f"<s>[INST] {question} [/INST]"
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        generation_kwargs = {
            "max_new_tokens": 256,
            "repetition_penalty": 1.2,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "temperature": max(0.01, min(temperature, 1.0)),
            "do_sample": temperature > 0.0,
            "top_p": top_p,
            "top_k": top_k,
        }
        try:
            outputs = self.model.generate(**inputs, **generation_kwargs)
        except RuntimeError as e:
            if "CUDA error" in str(e):
                import torch as _torch
                _torch.cuda.empty_cache()
                try:
                    generation_kwargs.update({"temperature": 1.0, "do_sample": False, "top_p": None, "top_k": None})
                    outputs = self.model.generate(**inputs, **generation_kwargs)
                except RuntimeError as e2:
                    if "CUDA error" in str(e2):
                        return "Unable to generate answer due to CUDA error."
                    raise e2
            else:
                raise e
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        marker = "[/INST]"
        return full_response.split(marker, 1)[1].strip() if marker in full_response else full_response.strip()


class QwenModel(BaseModel):
    def __init__(self, model_name: str = "Qwen/Qwen3-8B", load_in_8bit: bool = False):  # or Qwen/Qwen3-32B
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_load_kwargs)

    def generate_answer(self, question: str, **kwargs) -> str:
        temperature = kwargs.get("temperature", 0.0)
        top_p = kwargs.get("top_p", 0.95)
        top_k = kwargs.get("top_k", 64)
        user_prompt = question
        if not user_prompt.lstrip().startswith("/no_think"):
            user_prompt = f"/no_think\n\n{user_prompt}"
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generation_kwargs = {
            "max_new_tokens": 256,
            "repetition_penalty": 1.2,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "temperature": max(0.01, min(temperature, 1.0)),
            "do_sample": temperature > 0.0,
            "top_p": top_p,
            "top_k": top_k,
        }
        try:
            generated_ids = self.model.generate(**model_inputs, **generation_kwargs)
        except RuntimeError as e:
            if "CUDA error" in str(e):
                import torch as _torch
                _torch.cuda.empty_cache()
                try:
                    generation_kwargs.update({"temperature": 1.0, "do_sample": False, "top_p": None, "top_k": None})
                    generated_ids = self.model.generate(**model_inputs, **generation_kwargs)
                except RuntimeError as e2:
                    if "CUDA error" in str(e2):
                        return "Unable to generate answer due to CUDA error."
                    raise e2
            else:
                raise e
        # Remove prompt portion
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        # Fix Qwen3 spacing if needed
        try:
            from .utils import fix_qwen3_spacing
            if "qwen3" in self.model_name.lower() or ("qwen" in self.model_name.lower() and "3" in self.model_name.lower()):
                response = fix_qwen3_spacing(response)
        except Exception:
            pass
        return response


class OLMoModel(BaseModel):
    def __init__(self, model_name: str = "allenai/OLMo-2-1124-7B", load_in_8bit: bool = False):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_load_kwargs)

    def generate_answer(self, question: str, **kwargs) -> str:
        temperature = kwargs.get("temperature", 0.0)
        top_p = kwargs.get("top_p", 0.95)
        top_k = kwargs.get("top_k", 64)
        formatted_prompt = f"User: {question}"
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        generation_kwargs = {
            "max_new_tokens": 256,
            "repetition_penalty": 1.2,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "temperature": max(0.01, min(temperature, 1.0)),
            "do_sample": temperature > 0.0,
            "top_p": top_p,
            "top_k": top_k,
        }
        try:
            outputs = self.model.generate(**inputs, **generation_kwargs)
        except RuntimeError as e:
            if "CUDA error" in str(e):
                import torch as _torch
                _torch.cuda.empty_cache()
                try:
                    generation_kwargs.update({"do_sample": False})
                    outputs = self.model.generate(**inputs, **generation_kwargs)
                except RuntimeError as e2:
                    if "CUDA error" in str(e2):
                        return "Unable to generate answer due to CUDA error."
                    raise e2
            else:
                raise e
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_response.replace(formatted_prompt, "").strip()


class GemmaModel(BaseModel):
    def __init__(self, model_name: str = "google/gemma-3-4b-it", load_in_8bit: bool = False):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_load_kwargs)

    def generate_answer(self, question: str, **kwargs) -> str:
        temperature = kwargs.get("temperature", 0.0)
        top_p = kwargs.get("top_p", 0.95)
        top_k = kwargs.get("top_k", 64)
        messages = [{"role": "user", "content": question}]
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        generation_kwargs = {
            "max_new_tokens": 256,
            "repetition_penalty": 1.2,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "temperature": max(0.01, min(temperature, 1.0)),
            "do_sample": temperature > 0.0,
            "top_p": top_p,
            "top_k": top_k,
        }
        try:
            outputs = self.model.generate(**inputs, **generation_kwargs)
        except RuntimeError as e:
            if "CUDA error" in str(e):
                import torch as _torch
                _torch.cuda.empty_cache()
                try:
                    generation_kwargs.update({"do_sample": False})
                    outputs = self.model.generate(**inputs, **generation_kwargs)
                except RuntimeError as e2:
                    if "CUDA error" in str(e2):
                        return "Unable to generate answer due to CUDA error."
                    raise e2
            else:
                raise e
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        marker = "<start_of_turn>model\n"
        return full_response.split(marker, 1)[1].strip() if marker in full_response else full_response.strip()


class GeminiVertexModel(BaseModel):
    """Native Vertex AI Gemini model using REST API with proper generationConfig."""

    def __init__(self, model_name: str = "google/gemini-2.5-flash", max_retries: int = 3):
        from google.auth import default
        from google.auth.transport.requests import Request

        self.model_name = model_name
        self._request = Request()
        self.max_retries = max(1, max_retries)
        self.default_max_tokens = int(os.environ.get("VERTEX_MAX_TOKENS", "2048"))
        self.location = os.environ.get("VERTEX_LOCATION", os.environ.get("GOOGLE_CLOUD_REGION", "us-central1"))
        self.project_id = os.environ.get("VERTEX_PROJECT_ID", os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
        if not self.project_id:
            raise RuntimeError("Set VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT for GeminiVertexModel.")
        cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if cred_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
        credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self.credentials = credentials
        self._refresh_credentials()

    def _refresh_credentials(self) -> None:
        """Refresh credentials if needed."""
        if not self.credentials.valid or self.credentials.expired:
            self.credentials.refresh(self._request)

    def _get_model_id(self) -> str:
        """Extract model ID from model_name (e.g., 'google/gemini-2.5-pro' -> 'gemini-2.5-pro')."""
        if "/" in self.model_name:
            return self.model_name.split("/", 1)[1]
        return self.model_name

    def _build_request_body(
        self,
        question: str,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build native Vertex AI request body with proper generationConfig."""
        system_prompt = kwargs.get("system")
        user_content = kwargs.get("user") or question

        # Build contents in Vertex format
        contents = []
        if "messages" in kwargs and kwargs["messages"]:
            for msg in kwargs["messages"]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    # System goes in systemInstruction, skip here
                    continue
                vertex_role = "model" if role == "assistant" else "user"
                contents.append({"role": vertex_role, "parts": [{"text": content}]})
        else:
            contents.append({"role": "user", "parts": [{"text": user_content}]})

        # Get generation params
        temperature = max(0.0, kwargs.get("temperature", 0.0))
        max_tokens = (
            kwargs.get("max_new_tokens")
            or kwargs.get("max_tokens")
            or self.default_max_tokens
        )
        top_p = kwargs.get("top_p", 0.95)
        top_k = kwargs.get("top_k", 64)

        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "topK": top_k,
                "maxOutputTokens": max_tokens,
            },
        }

        # Add system instruction if present
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        elif "messages" in kwargs and kwargs["messages"]:
            # Check for system message in messages list
            for msg in kwargs["messages"]:
                if msg.get("role") == "system":
                    body["systemInstruction"] = {"parts": [{"text": msg.get("content", "")}]}
                    break

        # Add stop sequences if present
        stop = kwargs.get("stop")
        if stop:
            if isinstance(stop, str):
                body["generationConfig"]["stopSequences"] = [stop]
            elif isinstance(stop, (list, tuple)):
                body["generationConfig"]["stopSequences"] = list(stop)

        return body

    def _extract_text_from_response(self, response: Dict[str, Any]) -> str:
        """Extract text from native Vertex AI response."""
        candidates = response.get("candidates", [])
        if not candidates:
            return ""
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            return ""
        return parts[0].get("text", "").strip()

    def _extract_usage_from_response(self, response: Dict[str, Any]) -> Dict[str, int]:
        """Extract token usage from response."""
        usage = response.get("usageMetadata", {})
        return {
            "prompt_tokens": usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
        }

    def generate_answer(self, question: str, **kwargs) -> str:
        usage_meta = kwargs.get("usage_meta") or {}
        if "usage_meta" in kwargs:
            kwargs = dict(kwargs)
            kwargs.pop("usage_meta", None)

        last_error: Optional[Exception] = None
        text: str = ""

        model_id = self._get_model_id()
        url = (
            f"https://{self.location}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{self.location}/"
            f"publishers/google/models/{model_id}:generateContent"
        )

        body = self._build_request_body(question, kwargs)
        gen_config = body.get("generationConfig", {})

        for attempt in range(1, self.max_retries + 1):
            try:
                self._refresh_credentials()

                data = json.dumps(body).encode("utf-8")
                req = urllib.request.Request(url, data=data, method="POST")
                req.add_header("Authorization", f"Bearer {self.credentials.token}")
                req.add_header("Content-Type", "application/json")

                with urllib.request.urlopen(req, timeout=300) as resp:
                    response = json.loads(resp.read().decode("utf-8"))

                text = self._extract_text_from_response(response)
                usage = self._extract_usage_from_response(response)

                if text:
                    _log_usage(
                        model_name=self.model_name,
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                        meta={
                            "context": "generation",
                            "backend": "vertex-gemini-native",
                            "attempt": attempt,
                            "success": True,
                            "temperature": gen_config.get("temperature", 0.0),
                            "top_p": gen_config.get("topP", 0.95),
                            "top_k": gen_config.get("topK", 64),
                            "max_output_tokens": gen_config.get("maxOutputTokens"),
                            **(usage_meta if isinstance(usage_meta, dict) else {}),
                        },
                    )
                    return text

                print(f"⚠️  Gemini empty response (attempt {attempt}/{self.max_retries}). Retrying...")

            except urllib.error.HTTPError as exc:
                last_error = exc
                error_body = exc.read().decode("utf-8") if exc.fp else ""
                print(f"⚠️  Gemini HTTP error {exc.code} (attempt {attempt}/{self.max_retries}): {error_body[:200]}")
            except Exception as exc:
                last_error = exc
                print(f"⚠️  Gemini API error (attempt {attempt}/{self.max_retries}): {exc}")

        if last_error:
            print(f"⚠️  Gemini exhausted retries: {last_error}")

        _log_usage(
            model_name=self.model_name,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=None,
            meta={"context": "generation", "backend": "vertex-gemini-native", "success": False},
        )
        return text


class VertexBatchHelper:
    """Helpers for Vertex batch prediction jobs (Gemini)."""

    @staticmethod
    def build_request(
        user_prompt: str,
        system_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> Dict[str, Any]:
        request: Dict[str, Any] = {
            "contents": [
                {"role": "user", "parts": [{"text": user_prompt}]},
            ],
            "generationConfig": {
                "temperature": max(0.0, temperature),
                "topP": top_p,
                "topK": top_k,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_prompt:
            request["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        return {"request": request}

    @staticmethod
    def write_jsonl(path: Path, lines: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            for line in lines:
                fp.write(json.dumps(line, ensure_ascii=False) + "\n")

    @staticmethod
    def upload_to_gcs(local_path: Path, gcs_uri: str) -> None:
        if not gcs_uri.startswith("gs://"):
            raise ValueError("Batch input URI must be a gs:// path.")
        gsutil = shutil.which("gsutil")
        if not gsutil:
            raise RuntimeError("gsutil not found on PATH. Upload the JSONL to GCS manually.")
        subprocess.run([gsutil, "cp", str(local_path), gcs_uri], check=True)

    @staticmethod
    def submit_batch_job(
        model_name: str,
        input_uri: str,
        output_uri_prefix: str,
        display_name: str,
        *,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
    ) -> str:
        from google.auth import default
        from google.auth.transport.requests import Request

        resolved_location = location or os.environ.get("VERTEX_LOCATION", os.environ.get("GOOGLE_CLOUD_REGION", "global"))
        resolved_project = project_id or os.environ.get("VERTEX_PROJECT_ID", os.environ.get("GOOGLE_CLOUD_PROJECT"))
        if not resolved_project:
            raise RuntimeError("Set VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT for batch jobs.")

        model_path = model_name
        if model_name.startswith("google/"):
            model_path = f"publishers/google/models/{model_name.split('/', 1)[1]}"

        if resolved_location == "global":
            base = "https://aiplatform.googleapis.com"
        else:
            base = f"https://{resolved_location}-aiplatform.googleapis.com"

        url = f"{base}/v1/projects/{resolved_project}/locations/{resolved_location}/batchPredictionJobs"

        body = {
            "displayName": display_name,
            "model": model_path,
            "inputConfig": {
                "instancesFormat": "jsonl",
                "gcsSource": {"uris": [input_uri]},
            },
            "outputConfig": {
                "predictionsFormat": "jsonl",
                "gcsDestination": {"outputUriPrefix": output_uri_prefix},
            },
        }

        credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(Request())
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Authorization", f"Bearer {credentials.token}")
        req.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(req) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        name = payload.get("name", "")
        if not name:
            raise RuntimeError(f"Batch job creation failed: {payload}")
        return name

    @staticmethod
    def list_gcs_jsonl(prefix: str) -> List[str]:
        if not prefix.startswith("gs://"):
            raise ValueError("Batch output URI must be a gs:// path.")
        gsutil = shutil.which("gsutil")
        if not gsutil:
            raise RuntimeError("gsutil not found on PATH. Download batch output manually.")
        result = subprocess.run([gsutil, "ls", prefix], check=True, capture_output=True, text=True)
        return [line.strip() for line in result.stdout.splitlines() if line.strip().endswith(".jsonl")]

    @staticmethod
    def read_gcs_jsonl(uri: str) -> List[str]:
        gsutil = shutil.which("gsutil")
        if not gsutil:
            raise RuntimeError("gsutil not found on PATH. Download batch output manually.")
        result = subprocess.run([gsutil, "cat", uri], check=True, capture_output=True, text=True)
        return [line for line in result.stdout.splitlines() if line.strip()]

    @staticmethod
    def extract_text(record: Dict[str, Any]) -> str:
        candidate = None
        response = record.get("response")
        if isinstance(response, list) and response:
            response = response[0]
        if isinstance(response, dict):
            candidate = response.get("candidates") or response.get("prediction") or response.get("predictions")
        elif record.get("predictions"):
            candidate = record.get("predictions")

        if isinstance(candidate, list) and candidate:
            candidate = candidate[0]
        if isinstance(candidate, dict):
            content = candidate.get("content")
            if isinstance(content, dict):
                parts = content.get("parts", [])
                if isinstance(parts, list) and parts:
                    text = parts[0].get("text")
                    if text:
                        return str(text).strip()
            if "text" in candidate:
                return str(candidate.get("text", "")).strip()
        return ""

class HFChatModel(BaseModel):
    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        if not HF_AVAILABLE:
            raise ImportError("Transformers not available. Install transformers + torch.")
        self.model_name = model_name
        
        # Use cache_dir from env if not provided
        if cache_dir is None:
            cache_dir = os.environ.get("HF_CACHE_DIR", "/mnt/shared/shared_hf_home/hub")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        load_args: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
            "cache_dir": cache_dir
        }
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_args)

    def apply_chat_template(self, user: str, system: Optional[str] = None) -> str:
        messages: List[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prefix = (system + "\n\n") if system else ""
            return prefix + user

    def generate_answer(self, question: str, **kwargs) -> str:
        """Generate answer using HuggingFace model.
        
        Args:
            question: The question/prompt text (can be user message only)
            **kwargs: Optional parameters
                - system: System prompt/instructions
                - temperature: Sampling temperature (default: 0.0)
                - top_p: Nucleus sampling parameter (default: 0.95)
                - top_k: Top-k sampling parameter (default: 64)
                - max_new_tokens: Maximum tokens to generate (default: 512)
        """
        system = kwargs.get("system")
        temperature = kwargs.get("temperature", 0.0)
        top_p = kwargs.get("top_p", 0.95)
        top_k = kwargs.get("top_k", 64)
        max_new_tokens = kwargs.get("max_new_tokens", 1024)
        
        # Apply chat template with system prompt if provided
        prompt = self.apply_chat_template(question, system)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Ensure temperature is valid (slightly above 0 for stability)
        if temperature <= 0.0:
            temperature = 0.01
        temperature = max(0.01, min(temperature, 1.0))
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": kwargs.get("repetition_penalty", 1.2),
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "temperature": temperature,
            "do_sample": temperature > 0.01,
            "top_p": top_p,
            "top_k": top_k,
        }
        
        try:
            outputs = self.model.generate(**inputs, **gen_kwargs)
        except RuntimeError as e:
            if "CUDA error" in str(e) or "out of memory" in str(e).lower():
                print(f"⚠️  CUDA error, attempting recovery...")
                import torch
                torch.cuda.empty_cache()
                try:
                    # Retry with reduced batch or no sampling
                    gen_kwargs["do_sample"] = False
                    outputs = self.model.generate(**inputs, **gen_kwargs)
                except RuntimeError as e2:
                    print(f"⚠️  HuggingFace generation error: {e2}")
                    return "Error: Unable to generate answer"
            else:
                raise e
        
        # Decode and extract only the generated portion
        generated_ids = outputs[0][len(inputs["input_ids"][0]):]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Clean thinking tags if present
        import re
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        text = re.sub(r'<thinking>.*?</thinking>\s*', '', text, flags=re.DOTALL)
        text = re.sub(r'\n\s*\n', '\n', text)
        prompt_tokens = inputs["input_ids"].shape[-1]
        completion_tokens = generated_ids.shape[-1] if hasattr(generated_ids, "shape") else len(generated_ids)
        _log_usage(
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            meta={"context": "generation", "backend": "hf-chat", **(usage_meta if isinstance(usage_meta, dict) else {})},
        )
        return text.strip()

class VLLMModel(BaseModel):
    """Wrapper that talks to a vLLM OpenAI-compatible server."""

    def __init__(
        self,
        model_name: str,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        disable_thinking: bool = True,
        tensor_parallel_size: int = 1,
        **_: Any,
    ):
        del tensor_parallel_size  # kept for backwards compatibility with callers
        self.model_name = model_name
        self.base_url = (
            base_url
            or os.environ.get("VLLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "http://localhost:8000/v1"
        )
        self.api_key = api_key or os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.disable_thinking = disable_thinking
        # Set a generous timeout for large model inference (10 minutes for long generations)
        timeout = httpx.Timeout(600.0, connect=60.0)
        # Disable HTTP/2 to avoid hanging with vLLM servers (use HTTP/1.1 only)
        http_client = httpx.Client(timeout=timeout, http2=False)
        self.client = openai.OpenAI(
            base_url=self.base_url, 
            api_key=self.api_key, 
            default_headers=default_headers,
            http_client=http_client
        )

    def _is_qwen(self) -> bool:
        lower = self.model_name.lower()
        return "qwen" in lower and "3" in lower

    def _supports_thinking_flag(self) -> bool:
        """Return True for Qwen variants that support chat_template_kwargs.enable_thinking.

        Known to be supported by Qwen3 and Qwen2.5 series when served via vLLM.
        """
        lower = self.model_name.lower()
        return ("qwen" in lower) and ("3" in lower or "2.5" in lower)

    def _needs_qwen_spacing_fix(self) -> bool:
        """Qwen3 specific spacing artifact observed with some vLLM versions.

        Keep the fix scoped to Qwen3 to avoid over-correcting other models.
        """
        lower = self.model_name.lower()
        return "qwen3" in lower

    def _build_messages(self, question: str, kwargs: Dict[str, Any]) -> List[Dict[str, str]]:
        if "messages" in kwargs and kwargs["messages"]:
            return kwargs["messages"]

        system_prompt = kwargs.get("system")
        user_content = kwargs.get("user") or question

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _build_extra_body(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        extra_body = dict(kwargs.get("extra_body", {}))

        top_k = kwargs.get("top_k", 64)
        if top_k is not None:
            extra_body.setdefault("top_k", top_k)

        repetition_penalty = kwargs.get("repetition_penalty", 1.2)
        if repetition_penalty is not None:
            extra_body.setdefault("repetition_penalty", repetition_penalty)

        presence_penalty = kwargs.get("presence_penalty")
        if presence_penalty is not None:
            extra_body.setdefault("presence_penalty", presence_penalty)

        frequency_penalty = kwargs.get("frequency_penalty")
        if frequency_penalty is not None:
            extra_body.setdefault("frequency_penalty", frequency_penalty)

        if self._supports_thinking_flag() and self.disable_thinking:
            chat_kwargs = dict(extra_body.get("chat_template_kwargs", {}))
            chat_kwargs.setdefault("enable_thinking", False)
            extra_body["chat_template_kwargs"] = chat_kwargs

        return extra_body

    def generate_answer(self, question: str, **kwargs) -> str:
        usage_meta = kwargs.get("usage_meta") or {}
        if "usage_meta" in kwargs:
            kwargs = dict(kwargs)
            kwargs.pop("usage_meta", None)
        temperature = max(0.0, kwargs.get("temperature", 0.0))
        max_tokens = kwargs.get("max_new_tokens", kwargs.get("max_tokens", 1024))

        prompt_override = kwargs.get("prompt")
        if prompt_override is not None:
            if isinstance(prompt_override, str):
                messages = [{"role": "user", "content": prompt_override}]
            else:
                messages = prompt_override
        else:
            messages = self._build_messages(question, kwargs)
        extra_body = self._build_extra_body(kwargs)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=max(0.01, min(temperature, 2.0)),
            top_p=kwargs.get("top_p", 0.95),
            max_tokens=max_tokens,
            stop=kwargs.get("stop"),
            extra_body=extra_body or None,
        )

        result = response.choices[0].message.content or ""
        try:
            from .utils import fix_qwen3_spacing
            if self._needs_qwen_spacing_fix():
                result = fix_qwen3_spacing(result)
        except Exception:
            pass
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None
        if prompt_tokens is None:
            prompt_tokens = len(" ".join(m["content"] for m in messages)) // 4
        if completion_tokens is None:
            completion_tokens = len(result) // 4
        _log_usage(
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            meta={"context": "generation", "backend": "vllm", **(usage_meta if isinstance(usage_meta, dict) else {})},
        )
        return result.strip()


def get_model(model_choice: str, model_name: Optional[str] = None) -> BaseModel:
    mc = (model_choice or "gemini").lower()
    if mc == "gemini":
        return GeminiVertexModel(model_name or "google/gemini-2.5-flash")
    if mc in ("qwen", "qwen3-8b", "qwen-3-8b", "qwen3-32b", "qwen-3-32b", "llama", "gemma", "mistral", "olmo"):
        name = model_name or {
            "qwen": "Qwen/Qwen3-8B",
            "qwen3-8b": "Qwen/Qwen3-8B",
            "qwen-3-8b": "Qwen/Qwen3-8B",
            "qwen3-32b": "Qwen/Qwen3-32B",
            "qwen-3-32b": "Qwen/Qwen3-32B",
            "llama": "meta-llama/Llama-3.1-8B-Instruct",
            "gemma": "google/gemma-3-4b-it",
            "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
            "olmo": "allenai/OLMo-2-1124-7B",
        }[mc]
        return HFChatModel(name)
    raise ValueError(f"Unknown model_choice: {model_choice}")


# Backwards compatibility aliases
VertexModel = GeminiVertexModel
GeminiModel = GeminiVertexModel

# ---------------------------------------------------------------------
# Token usage logging helper
# ---------------------------------------------------------------------

_TOKEN_LOG_PATH = Path(os.environ.get("TOKEN_LOG_PATH", "results/token_usage_log.jsonl"))


def _log_usage(
    *,
    model_name: str,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    total_tokens: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a token-usage record; non-fatal on error."""
    if os.environ.get("TOKEN_LOG_DISABLE"):
        return
    try:
        pt = int(prompt_tokens or 0)
        ct = int(completion_tokens or 0)
        tt = int(total_tokens) if total_tokens is not None else pt + ct
        _TOKEN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model": model_name,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": tt,
            "meta": meta or {},
        }
        with _TOKEN_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        return


# ---------------------------------------------------------------------
# GPT-5.1 Batch helpers
# ---------------------------------------------------------------------

def build_gpt51_batch_line(
    *,
    custom_id: str,
    question: str,
    model_name: str = "gpt-5.1",
    reasoning_effort: Optional[str] = None,
    max_completion_tokens: Optional[int] = None,
    stop: Optional[Iterable[str]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    system: Optional[str] = None,
    user: Optional[str] = None,
) -> str:
    """Construct one JSONL line for a Chat Completions batch job.

    Follows the same parameter gating rules as GPT51Model:
    - When reasoning_effort != "none", temperature/top_p are omitted.
    - Uses max_completion_tokens for output cap.
    - Batch must point to /v1/chat/completions and contain only independent requests.
    """
    model = GPT51Model(model_name=model_name)  # lightweight; only for body construction
    body = model._build_body(  # type: ignore[attr-defined]
        question,
        {
            "messages": messages,
            "system": system,
            "user": user,
            "reasoning_effort": reasoning_effort,
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": GPT51Model._normalize_stop(stop) if stop is not None else None,
            "model_name": model_name,
        },
    )
    return json.dumps(
        {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
    )


def parse_gpt51_batch_output(jsonl_text: str) -> Dict[str, str]:
    """Parse Batch output JSONL into {custom_id: assistant_text}.

    Handles completed items only; ignores lines without a response or with errors.
    """
    results: Dict[str, str] = {}
    for raw_line in jsonl_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        custom_id = payload.get("custom_id") or payload.get("id") or None
        if not custom_id:
            continue
        response = payload.get("response", {})
        body = response.get("body") if isinstance(response, dict) else {}
        choices = body.get("choices") if isinstance(body, dict) else None
        if not choices:
            continue
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if message and isinstance(message, dict):
            text = message.get("content") or ""
            results[str(custom_id)] = text
    return results


def parse_gpt51_batch_output_with_usage(jsonl_text: str) -> Dict[str, Dict[str, Any]]:
    """Parse Batch output JSONL into {custom_id: {content, usage, model}}."""
    results: Dict[str, Dict[str, Any]] = {}
    for raw_line in jsonl_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        custom_id = payload.get("custom_id") or payload.get("id") or None
        if not custom_id:
            continue
        response = payload.get("response", {})
        body = response.get("body") if isinstance(response, dict) else {}
        choices = body.get("choices") if isinstance(body, dict) else None
        usage = body.get("usage") if isinstance(body, dict) else {}
        model = body.get("model") if isinstance(body, dict) else None
        if not choices:
            continue
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if message and isinstance(message, dict):
            text = message.get("content") or ""
            if not text:
                _warn_empty_completion(
                    content=text,
                    usage=usage,
                    model_name=model,
                    example_id=str(custom_id).split("::", 1)[0] if "::" in str(custom_id) else None,
                    regime=str(custom_id).split("::", 1)[1] if "::" in str(custom_id) else None,
                    context="batch",
                )
            results[str(custom_id)] = {"content": text, "usage": usage or {}, "model": model}
    return results


def _warn_empty_completion(
    *,
    content: str,
    usage: Any,
    model_name: Optional[str],
    example_id: Optional[str] = None,
    regime: Optional[str] = None,
    context: str = "sync",
) -> None:
    """Emit a warning when completion content is empty, especially if reasoning consumed the budget."""
    if content:
        return
    usage = usage or {}
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    details = usage.get("completion_tokens_details", {}) if isinstance(usage, dict) else {}
    reasoning_tokens = details.get("reasoning_tokens")
    hint = ""
    if completion_tokens and reasoning_tokens == completion_tokens:
        hint = " (reasoning tokens consumed entire completion budget)"
    prefix = "⚠️"
    parts = [p for p in [example_id, regime] if p]
    target = "::".join(parts) if parts else None
    label = f"{target} " if target else ""
    model_label = f"{model_name} " if model_name else ""
    print(f"{prefix} Empty completion {label}via {context} on {model_label}{hint}.")
