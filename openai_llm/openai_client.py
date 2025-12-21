import logging
import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from openai import OpenAI


@dataclass
class OpenAIConfig:
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    enable_retry: bool = True  # Enable retry with exponential backoff
    max_retries: int = 5  # Maximum retry attempts per request
    initial_retry_delay: float = 1.0  # Initial delay in seconds
    max_retry_delay: float = 60.0  # Maximum delay in seconds


@dataclass
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2
    n: int = 1
    seed: Optional[int] = None


class LLMClient:
    """
    LLM client using OpenAI-compatible API.
    Maintains the same interface as the vLLM-based client.
    """

    def __init__(
        self,
        model_name: str,
        config: OpenAIConfig,
    ):
        self.model_name = model_name
        self.config = config

        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

        logging.info(
            f"OpenAI LLMClient initialized for model={self.model_name} ")

    def _load_existing_results(self, output_path: Optional[str]) -> List[Dict[str, Any]]:
        """Load existing results from output file to resume from where we left off."""
        if not output_path:
            return []

        out_file = Path(output_path)
        if not out_file.exists():
            return []

        try:
            with out_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                logging.info(f"Resuming: loaded {len(data)} existing results from output file")
                return data
            else:
                logging.warning(f"Output file exists but has unexpected structure. Starting fresh.")
                return []
        except Exception as e:
            logging.warning(f"Could not load existing results: {e}. Starting fresh.")
            return []

    def _save_results(self, output_path: Optional[str], results: List[Dict[str, Any]]):
        """Save results to output file (acts as checkpoint for resume)."""
        if not output_path:
            return

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with out_file.open("w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved {len(results)} results")
        except Exception as e:
            logging.warning(f"Failed to save results: {e}")

    def _is_quota_error(self, error: Exception) -> bool:
        """Check if error is a quota/rate limit error that should stop processing."""
        error_str = str(error).lower()
        # Gemini quota errors
        if "quota" in error_str or "resource_exhausted" in error_str:
            return True
        # OpenAI rate limit errors that indicate daily quota
        if "insufficient_quota" in error_str or "billing" in error_str:
            return True
        return False

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is retryable (temporary issue)."""
        error_str = str(error).lower()
        # Rate limit errors (temporary)
        if "rate_limit" in error_str or "429" in error_str:
            return True
        # Server errors (temporary)
        if "500" in error_str or "502" in error_str or "503" in error_str or "504" in error_str:
            return True
        # Timeout errors
        if "timeout" in error_str or "timed out" in error_str:
            return True
        return False

    def run_batch(
        self,
        prompts: List[Dict[str, Any]],
        sampling_params: SamplingConfig,
        output_field: str = "output",
        output_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of prompts and return results with metadata.

        Supports:
        - Resume from existing output file (no separate checkpoint needed)
        - Exponential backoff retry for transient errors
        - Graceful stop on quota exhaustion
        - Filters out [ERROR] responses from final output

        When n > 1, returns a list of responses in the output_field.
        """
        # Load existing results to resume
        results = self._load_existing_results(output_path)

        # Build set of completed (persona_id, qid, variant, seed) tuples
        completed = set()
        for r in results:
            key = (r.get("persona_id"), r.get("question_id"), r.get("variant"), r.get("seed"))
            completed.add(key)

        quota_exhausted = False

        for idx, prompt in enumerate(prompts):
            # Check if this prompt already completed
            meta = prompt.get("metadata", {})
            key = (meta.get("persona_id"), meta.get("qid"), meta.get("variant"), sampling_params.seed)
            if key in completed:
                continue

            if quota_exhausted:
                logging.warning(f"Skipping remaining prompts due to quota exhaustion")
                break

            retry_count = 0
            success = False

            while retry_count <= self.config.max_retries and not success:
                try:
                    messages = prompt["messages"]

                    # Build API call parameters
                    api_params = {
                        "model": self.model_name,
                        "messages": messages,
                        "temperature": sampling_params.temperature,
                        "n": sampling_params.n,
                    }

                    # OpenAI uses max_completion_tokens, other providers use max_tokens
                    if "api.openai.com" in self.config.base_url:
                        api_params["max_completion_tokens"] = sampling_params.max_tokens
                        api_params["reasoning_effort"] = "none"
                    else:
                        api_params["max_tokens"] = sampling_params.max_tokens

                    # Add seed if provided (DeepSeek and Google don't support it, so skip)
                    if sampling_params.seed is not None and not ("deepseek.com" in self.config.base_url or "google" in self.config.base_url):
                        api_params["seed"] = sampling_params.seed

                    # DeepSeek-specific: disable thinking mode
                    if "deepseek.com" in self.config.base_url:
                        api_params["extra_body"] = {"thinking": {"type": "disabled"}}

                    if "gpt-oss" in self.model_name:
                        api_params["extra_body"] = {"reasoning_effort": "low"}

                    if "anthropic.com" not in self.config.base_url:
                        api_params["top_p"] = sampling_params.top_p

                    response = self.client.chat.completions.create(**api_params)

                    # Handle n > 1: return list of responses
                    if sampling_params.n > 1:
                        outputs = [choice.message.content.strip() for choice in response.choices]
                        results.append({
                            **prompt.get("metadata", {}),
                            output_field: outputs,
                        })
                    else:
                        output = response.choices[0].message.content.strip()
                        results.append({
                            **prompt.get("metadata", {}),
                            output_field: output,
                        })

                    # Mark as completed
                    completed.add(key)
                    success = True

                    # Save results periodically (every 10 successful requests)
                    if len(results) % 10 == 0:
                        self._save_results(output_path, results)

                    # Log progress
                    if len(results) % 10 == 0:
                        logging.info(f"Progress: {len(results)} prompts completed")

                except Exception as e:
                    error_msg = str(e)

                    # Check if quota exhausted (should stop)
                    if self._is_quota_error(e):
                        logging.error(f"Quota exhausted: {error_msg}")
                        logging.error(f"Processed {len(results)}/{len(prompts)} prompts before quota exhaustion")
                        logging.error(f"Results saved. Rerun same command to resume.")
                        quota_exhausted = True
                        self._save_results(output_path, results)
                        break

                    # Check if retryable
                    if self.config.enable_retry and self._is_retryable_error(e) and retry_count < self.config.max_retries:
                        retry_count += 1
                        delay = min(
                            self.config.initial_retry_delay * (2 ** (retry_count - 1)),
                            self.config.max_retry_delay
                        )
                        logging.warning(f"Retryable error (attempt {retry_count}/{self.config.max_retries}): {error_msg}")
                        logging.warning(f"Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        # Non-retryable error or max retries exceeded - log but don't add to results
                        logging.error(f"Failed after {retry_count} retries: {error_msg}")
                        # Mark as completed to avoid infinite retries on resume
                        completed.add(key)
                        # Don't add error responses to results (they'll be filtered out)
                        break

        # Final save
        self._save_results(output_path, results)

        # Filter out any [ERROR] responses that might have slipped through
        filtered_results = []
        for r in results:
            output = r.get(output_field, "")
            if isinstance(output, list):
                # For n > 1, filter out error strings
                clean_outputs = [o for o in output if not str(o).startswith("[ERROR]")]
                if clean_outputs:
                    r[output_field] = clean_outputs
                    filtered_results.append(r)
            else:
                # For n = 1, skip error responses
                if not str(output).startswith("[ERROR]"):
                    filtered_results.append(r)

        logging.info(f"Completed: {len(filtered_results)}/{len(prompts)} prompts successfully processed")
        if quota_exhausted:
            logging.warning(f"Stopped due to quota exhaustion. Resume with checkpoint to continue.")

        return filtered_results

    def delete_client(self):
        """Delete the OpenAI client."""
        logging.info("Deleting OpenAI client")
        self.client = None
        logging.info("OpenAI client deleted")

    def reset_client_to_another_model(self, model_name: str):
        """Reset the client to use a different model."""
        logging.info(f"Resetting OpenAI client to model {model_name}")
        self.model_name = model_name
        # Recreate client if needed, but since model is in create, it's fine
        logging.info(f"OpenAI client reset to model: {self.model_name}")
