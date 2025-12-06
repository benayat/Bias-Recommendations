from llm.llm_client import LLMResourceConfig, SamplingConfig

# Configuration for home setup with RTX 3090 (single GPU, 24GB VRAM)
HOME_CONFIG = LLMResourceConfig(
    gpu_memory_utilization = 0.95,
    max_model_len=4096,
    max_num_seqs=256,
    max_num_batched_tokens=1048576,
    block_size=16,  # Standard KV cache block size
    tensor_parallel_size=1,  # Single GPU
    dtype="auto",  # Automatic data type selection
    trust_remote_code=True,  # Allow custom models
    disable_log_stats=True,  # Disable verbose logging
    max_parallel_loading_workers=4,  # Parallel loading for faster startup
    enable_prefix_caching=True,  # Enable prefix caching
    enforce_eager=False,  # Use default execution mode
    use_transformers=False,  # Use vLLM backend
)
HOME_CONFIG_SMALL = LLMResourceConfig(
    gpu_memory_utilization = 0.9,
    max_model_len=4096,  # Limited to ~250 expected tokens (input + output) for efficiency
    max_num_seqs=16,  # Moderate concurrency
    max_num_batched_tokens=65536,
    block_size=16,  # Standard KV cache block size
    tensor_parallel_size=1,  # Single GPU
    dtype="auto",  # Automatic data type selection
    trust_remote_code=True,  # Allow custom models
    disable_log_stats=True,  # Disable verbose logging
    max_parallel_loading_workers=4,  # Parallel loading for faster startup
    enable_prefix_caching=True,  # Enable prefix caching
    enforce_eager=False,  # Use default execution mode
    use_transformers=False,  # Use vLLM backend
)

HPC_2H200_CONFIG = LLMResourceConfig(
    gpu_memory_utilization=0.92,        # small headroom vs 0.95
    max_model_len=4096,
    max_num_seqs=64,                  # high concurrency, real limiter is KV
    max_num_batched_tokens=262144,   # 2^20, fits under ~1.36M KV tokens
    block_size=16,
    tensor_parallel_size=2,
    dtype="auto",
    trust_remote_code=True,
    disable_log_stats=True,
    max_parallel_loading_workers=16,
    enable_prefix_caching=True,
    enforce_eager=False,
    use_transformers=False,
)

DEFAULT_SAMPLING_CONFIG = SamplingConfig(
    temperature=0.0,  # Deterministic for estimation tasks
    top_p=1.0,  # No nucleus sampling
    max_tokens=8,  # Sufficient for compensation estimates
)
