# =============================================================================
# chunking.py
# =============================================================================
"""
本模块为 YourBench 流水线实现了两种分块（chunking）模式：
1) "fast_chunking"（新的默认方式），仅根据长度规则进行分块。
2) "semantic_chunking"（需要显式配置），利用句子嵌入和相似度阈值来决定分块边界。

用法说明：
------
通常不需要直接调用本模块。只要在流水线配置中启用 pipeline.chunking.run，handler.py 会自动调用 run(config)。

run(config) 函数主要功能如下：
1. 加载流水线配置中指定的数据集。
2. 根据配置的分块模式执行不同策略：
   - fast_chunking（默认）：仅基于最大 token 长度对文本进行分块，不考虑句子相似性。
   - semantic_chunking（需要配置 pipeline.chunking.chunking_configuration.chunking_mode="semantic_chunking"）：在用户定义的 token 长度限制（l_min_tokens, l_max_tokens）和相似度阈值（tau_threshold）指导下，将每个文档切分为单跳（single-hop）分块。
3. 通过对单跳分块采样并拼接，创建多跳（multi-hop）分块。
4. 如果启用调试模式（debug mode），为每个分块计算可读性（readability）和困惑度（perplexity）等可选指标。
5. 保存带有新列的数据集，包括：
   - "chunks" （单跳分段的列表）
   - "multihop_chunks" （多跳分段组的列表）
   - "chunk_info_metrics" （各种统计信息）
   - "chunking_model" （用于嵌入的模型名；若为 fast_chunking 则为空或默认）

错误处理与日志：
---------------------------
- 所有警告、错误及调试信息会同时记录到控制台和专用日志文件 logs/chunking.log。
- 在加载或处理数据过程中如有严重错误发生，系统会记录异常并尝试优雅地退出，避免整个流水线崩溃。

"""

import os
import re
import random
import itertools
from typing import Any, Dict, Optional
from dataclasses import asdict, dataclass

from loguru import logger

from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset


# Try importing torch-related libraries
_torch_available = False
try:
    import torch
    import torch.nn.functional as F
    from torch.amp import autocast

    _torch_available = True
    logger.info("PyTorch is available.")
except ImportError:
    logger.info("PyTorch is not available. Semantic chunking features requiring torch will be disabled.")

    # Define dummy autocast if torch not found
    class DummyAutocast:
        def __enter__(self):
            pass

        def __exit__(self, type, value, traceback):
            pass

    def autocast(device_type):
        return DummyAutocast()  # type: ignore


# Try importing transformers
_transformers_available = False
try:
    from transformers import AutoModel, AutoTokenizer

    _transformers_available = True
    logger.info("Transformers library is available.")
except ImportError:
    logger.info(
        "Transformers library is not available. Semantic chunking features requiring transformers will be disabled."
    )
    AutoModel = None  # type: ignore
    AutoTokenizer = None  # type: ignore


# Try importing tiktoken for token counting
_tiktoken_available = False
_tiktoken_encoding = None
try:
    import tiktoken
    # Using a common default encoder. Consider making this configurable.
    # Common options: "cl100k_base", "p50k_base", "r50k_base"
    _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
    _tiktoken_available = True
    logger.info("tiktoken library is available and encoder 'cl100k_base' loaded.")
except ImportError:
    logger.warning(
        "tiktoken library is not available. Token counting will fall back to space splitting. "
        "Install with 'pip install tiktoken'."
    )
except Exception as e:
    logger.warning(
        f"tiktoken encoder could not be loaded (using 'cl100k_base'). Token counting will fall back to space splitting. Error: {e}"
    )


try:
    import evaluate

    # Attempt to load perplexity metric from evaluate
    _perplexity_metric = evaluate.load("perplexity", module_type="metric", model_id="gpt2")
    logger.info("Loaded 'perplexity' metric with model_id='gpt2'.")
except Exception as perplexity_load_error:
    logger.info(
        f"Could not load perplexity metric from 'evaluate'. Skipping perplexity. Error: {perplexity_load_error}"
    )
    _perplexity_metric = None

try:
    # Attempt to import textstat for readability metrics
    import textstat

    _use_textstat = True
except ImportError:
    logger.info("Package 'textstat' not installed. Readability metrics will be skipped.")
    _use_textstat = False


# -----------------------------------------------------------------------------
# Dataclasses for cleaner configuration and result handling
# -----------------------------------------------------------------------------
@dataclass
class ChunkingParameters:
    l_min_tokens: int = 64
    l_max_tokens: int = 128
    tau_threshold: float = 0.3
    h_min: int = 2
    h_max: int = 3
    num_multihops_factor: int = 2
    chunking_mode: str = "fast_chunking"  # "fast_chunking" or "semantic_chunking"


@dataclass
class SingleHopChunk:
    chunk_id: str
    chunk_text: str


@dataclass
class MultiHopChunk:
    chunk_ids: list[str]
    chunks_text: list[str]


@dataclass
class ChunkInfoMetrics:
    token_count: float
    unique_token_ratio: float
    bigram_diversity: float
    perplexity: float
    avg_token_length: float
    flesch_reading_ease: float
    gunning_fog: float


def _parse_chunking_parameters(config: Dict[str, Any]) -> ChunkingParameters:
    """
    Extracts the chunking parameters from the config dictionary, falling back
    to default values if keys are missing. The chunking_mode defaults to
    "fast_chunking" unless explicitly set to "semantic_chunking."
    """
    chunking_params = config.get("pipeline", {}).get("chunking", {}).get("chunking_configuration", {})
    return ChunkingParameters(
        l_min_tokens=chunking_params.get("l_min_tokens", 64),
        l_max_tokens=chunking_params.get("l_max_tokens", 128),
        tau_threshold=chunking_params.get("tau_threshold", 0.3),
        h_min=chunking_params.get("h_min", 2),
        h_max=chunking_params.get("h_max", 3),
        num_multihops_factor=chunking_params.get("num_multihops_factor", 2),
        chunking_mode=chunking_params.get("chunking_mode", "fast_chunking"),
    )


def run(config: Dict[str, Any]) -> None:
    """
    Main pipeline entry point for the chunking stage.

    Args:
        config (Dict[str, Any]): The entire pipeline configuration dictionary.

    Returns:
        None. This function saves the updated dataset containing chunked
        documents to disk or the Hugging Face Hub, based on the config.

    Raises:
        RuntimeError: If a critical error is encountered that prevents chunking.
                      The error is logged, and execution attempts a graceful exit.
    """
    # Retrieve chunking configuration from config
    chunking_config = config.get("pipeline", {}).get("chunking", {})
    if chunking_config is None or not chunking_config.get("run", False):
        logger.info("Chunking stage is disabled. Skipping.")
        return

    logger.info("Starting chunking stage...")

    # Attempt to load dataset
    dataset = custom_load_dataset(config=config, subset="summarized")
    logger.info(f"Loaded summarized subset with {len(dataset)} rows for chunking.")

    # Retrieve chunking parameters into a dataclass
    params = _parse_chunking_parameters(config)
    l_min_tokens = params.l_min_tokens
    l_max_tokens = params.l_max_tokens
    tau_threshold = params.tau_threshold
    h_min = params.h_min
    h_max = params.h_max
    num_multihops_factor = params.num_multihops_factor
    chunking_mode = params.chunking_mode.lower().strip()

    # Check if tiktoken is available for token counting
    use_tiktoken_for_counting = _tiktoken_available and _tiktoken_encoding is not None
    if not use_tiktoken_for_counting:
        logger.warning("tiktoken not available or encoder failed to load. Using space splitting for token counts.")

    # Check debug setting
    debug_mode: bool = config.get("settings", {}).get("debug", False)
    if debug_mode is False:
        # If not debug mode, skip perplexity and readability to save time
        logger.debug("Skipping perplexity and readability metrics (debug mode off).")
        local_perplexity_metric = None
        local_use_textstat = False
    else:
        local_perplexity_metric = _perplexity_metric
        local_use_textstat = _use_textstat

    # We'll only load the chunking model if in semantic_chunking mode
    tokenizer = None
    model = None
    device = "cpu"
    model_name = "/home/tom/fssd/model/cache/models--BAAI--bge-m3"

    if chunking_mode == "semantic_chunking":
        # Check if required libraries are installed
        if not _torch_available or not _transformers_available:
            logger.error(
                "Semantic chunking requires 'torch' and 'transformers' libraries. "
                "Please install them (e.g., pip install yourbench[semantic]) or use 'fast_chunking' mode."
            )
            return  # Exit if dependencies are missing for semantic chunking

        try:
            # Extract model name from config if available
            model_name_list = config.get("model_roles", {}).get("chunking", [])
            if model_name_list is None or len(model_name_list) == 0:
                logger.info(
                    "No chunking model specified in config['model_roles']['chunking']. "
                    "Using default '/home/tom/fssd/model/cache/models--BAAI--bge-m3'."
                )
                model_name = "/home/tom/fssd/model/cache/models--BAAI--bge-m3"
            else:
                model_name = model_name_list[0]

            logger.info(f"Using chunking model: '{model_name}'")
            # Determine device only if torch is available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore
            model = AutoModel.from_pretrained(model_name).to(device).eval()  # type: ignore
        except Exception as model_error:
            logger.error(f"Error loading tokenizer/model '{model_name}': {model_error}")
            logger.warning("Chunking stage cannot proceed with semantic_chunking. Exiting.")
            return
    else:
        logger.info("Using fast_chunking mode: purely length-based chunking with no embeddings.")

    # Prepare data structures
    all_single_hop_chunks: list[list[SingleHopChunk]] = []
    all_multihop_chunks: list[list[MultiHopChunk]] = []
    all_chunk_info_metrics: list[list[ChunkInfoMetrics]] = []
    all_similarities: list[list[float]] = []

    # Process each document in the dataset
    for idx, row in enumerate(dataset):
        doc_text = row.get("document_text", "")
        doc_id = row.get("document_id", f"doc_{idx}")

        # If text is empty or missing
        if doc_text is None or not doc_text.strip():
            logger.warning(f"Document at index {idx} has empty text. Storing empty chunks.")
            all_single_hop_chunks.append([])
            all_multihop_chunks.append([])
            all_chunk_info_metrics.append([])
            continue

        # Split the document into sentences
        sentences = _split_into_sentences(doc_text)
        if sentences is None or len(sentences) == 0:
            logger.warning(f"No valid sentences found for doc at index {idx}, doc_id={doc_id}.")
            all_single_hop_chunks.append([])
            all_multihop_chunks.append([])
            all_chunk_info_metrics.append([])
            continue

        # Depending on the chunking mode:
        if chunking_mode == "semantic_chunking":
            # Ensure dependencies one last time before computation
            if not _torch_available or not _transformers_available or model is None or tokenizer is None:
                logger.error("Cannot perform semantic chunking due to missing dependencies or model loading issues.")
                # Add empty lists and continue to avoid crashing the loop for this document
                all_single_hop_chunks.append([])
                all_multihop_chunks.append([])
                all_chunk_info_metrics.append([])
                continue

            # 1) Compute embeddings for sentences
            sentence_embeddings = _compute_embeddings(tokenizer, model, texts=sentences, device=device, max_len=512)
            # 2) Compute consecutive sentence similarities
            consecutive_sims: list[float] = []
            for sentence_index in range(len(sentences) - 1):
                cos_sim = float(
                    F.cosine_similarity(
                        sentence_embeddings[sentence_index].unsqueeze(0),
                        sentence_embeddings[sentence_index + 1].unsqueeze(0),
                        dim=1,
                    )[0]
                )
                consecutive_sims.append(cos_sim)
            if consecutive_sims:
                all_similarities.append(consecutive_sims)

            # 3) Create single-hop chunks with semantic logic
            single_hop_chunks = _chunk_document_semantic(
                sentences=sentences,
                similarities=consecutive_sims,
                l_min_tokens=l_min_tokens,
                l_max_tokens=l_max_tokens,
                tau=tau_threshold,
                doc_id=doc_id,
                use_tiktoken=use_tiktoken_for_counting,
            )
        else:
            # Fast chunking: Choose based on tiktoken availability
            if use_tiktoken_for_counting and _tiktoken_encoding:
                logger.debug(f"Using improved fast chunking (tiktoken-based) for doc_id={doc_id}")
                single_hop_chunks = _chunk_document_fast_tiktoken(
                    text=doc_text,
                    l_min_tokens=l_min_tokens,
                    l_max_tokens=l_max_tokens,
                    doc_id=doc_id,
                    tokenizer=_tiktoken_encoding
                )
            else:
                logger.warning(
                    f"tiktoken not available or encoder failed. Using fallback fast chunking (sentence split + max length) for doc_id={doc_id}. "
                    "Improved fast chunking requires 'pip install tiktoken'."
                )
                # Fallback requires sentences
                sentences = _split_into_sentences(doc_text)
                if sentences is None or len(sentences) == 0:
                    logger.warning(f"No valid sentences found for doc at index {idx}, doc_id={doc_id} during fallback fast chunking.")
                    single_hop_chunks = []
                else:
                    single_hop_chunks = _chunk_document_fast_fallback(
                        sentences=sentences,
                        l_max_tokens=l_max_tokens,
                        doc_id=doc_id,
                        use_tiktoken=False, # Explicitly false for fallback
                    )

        # Create multi-hop chunks
        multihop = _multihop_chunking(
            single_hop_chunks,
            h_min=h_min,
            h_max=h_max,
            num_multihops_factor=num_multihops_factor,
        )

        # Compute metrics (token_count, perplexity, readability, etc.)
        chunk_metrics = _compute_info_density_metrics(
            single_hop_chunks,
            local_perplexity_metric,
            local_use_textstat,
            use_tiktoken=use_tiktoken_for_counting,
        )

        # Accumulate
        all_single_hop_chunks.append(single_hop_chunks)
        all_multihop_chunks.append(multihop)
        all_chunk_info_metrics.append(chunk_metrics)

    # Optional: Save aggregated similarity plot only if in semantic_chunking and debug
    if chunking_mode == "semantic_chunking" and all_similarities and debug_mode:
        _plot_aggregated_similarities(all_similarities)

    # Convert dataclasses back to dicts for safe addition to the dataset
    dataset = dataset.add_column(
        "chunks",
        [[asdict(chunk) for chunk in chunk_list] for chunk_list in all_single_hop_chunks],
    )
    dataset = dataset.add_column(
        "multihop_chunks",
        [[asdict(mh) for mh in multihop_list] for multihop_list in all_multihop_chunks],
    )
    dataset = dataset.add_column(
        "chunk_info_metrics",
        [[asdict(cm) for cm in metric_list] for metric_list in all_chunk_info_metrics],
    )
    dataset = dataset.add_column("chunking_model", [model_name] * len(dataset))

    # Save updated dataset
    custom_save_dataset(dataset=dataset, config=config, subset="chunked")
    logger.success("Chunking stage completed successfully.")


def _split_into_sentences(text: str) -> list[str]:
    """
    Splits the input text into sentences using a simple rule-based approach
    that looks for punctuation delimiters ('.', '!', '?').

    Args:
        text (str): The full document text to be split.

    Returns:
        list[str]: A list of sentence strings.
    """
    # Replace newlines with spaces for consistency
    normalized_text = text.replace("\n", " ").strip()
    if normalized_text is None or normalized_text == "":
        return []

    # Split using capturing parentheses to retain delimiters, then recombine.
    segments = re.split(r"([.!?])", normalized_text)
    sentences: list[str] = []
    for i in range(0, len(segments), 2):
        if i + 1 < len(segments):
            # Combine the text and delimiter
            candidate = (segments[i] + segments[i + 1]).strip()
        else:
            # If no delimiter segment, use the text directly
            candidate = segments[i].strip()
        if candidate:
            sentences.append(candidate)
    return sentences


def _compute_embeddings(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    texts: list[str],
    device: "torch.device",
    max_len: int = 512,
    batch_size: int = 16,
) -> "list[torch.Tensor]":
    """
    Computes sentence embeddings by mean pooling the last hidden states,
    normalized to unit length.

    Args:
        tokenizer (AutoTokenizer): A Hugging Face tokenizer.
        model (AutoModel): A pretrained transformer model to generate embeddings.
        texts (list[str]): The list of sentence strings to be embedded.
        device (torch.device): The device on which to run inference (CPU or GPU).
        max_len (int): Max sequence length for tokenization.
        batch_size (int): Batch size.
    Returns:
        list[torch.Tensor]: A list of PyTorch tensors (one per sentence).
    """
    embeddings = []
    model.eval()

    # Determine autocast device type string
    autocast_device_type = "cuda" if _torch_available and torch.cuda.is_available() else "cpu"

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_dict = tokenizer(batch_texts, max_length=max_len, padding=True, truncation=True, return_tensors="pt").to(
            device
        )

        with torch.no_grad():
            # Use autocast context manager
            with autocast(autocast_device_type):
                outputs = model(**batch_dict)
                last_hidden_states = outputs.last_hidden_state
                attention_mask = batch_dict["attention_mask"]

                # Zero out non-attended tokens
                last_hidden_states = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

                # Mean pooling
                sum_hidden = last_hidden_states.sum(dim=1)
                valid_token_counts = attention_mask.sum(dim=1, keepdim=True)
                batch_embeddings = sum_hidden / valid_token_counts.clamp(min=1e-9)

                # Normalize
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

        embeddings.extend(batch_embeddings.cpu())

    return embeddings


def _chunk_document_semantic(
    sentences: list[str],
    similarities: list[float],
    l_min_tokens: int,
    l_max_tokens: int,
    tau: float,
    doc_id: str,
    use_tiktoken: bool,
) -> list[SingleHopChunk]:
    """
    Creates single-hop chunks from sentences using semantic guidance. Ensures each
    chunk is at least l_min_tokens in length and at most l_max_tokens, introducing
    a chunk boundary when consecutive sentence similarity is below threshold tau.

    Args:
        sentences (list[str]): The list of sentences for a single document.
        similarities (list[float]): Cosine similarities between consecutive sentences.
        l_min_tokens (int): Minimum tokens per chunk.
        l_max_tokens (int): Maximum tokens per chunk.
        tau (float): Similarity threshold for introducing a chunk boundary.
        doc_id (str): Unique identifier for the document.
        use_tiktoken (bool): If True, use tiktoken for counting. Otherwise, split by space.

    Returns:
        list[SingleHopChunk]: A list of SingleHopChunk objects.
    """
    chunks: list[SingleHopChunk] = []
    current_chunk: list[str] = []
    current_len: int = 0
    chunk_index: int = 0

    for i, sentence in enumerate(sentences):
        # Calculate token count based on availability
        if use_tiktoken and _tiktoken_encoding:
            try:
                sentence_token_count = len(_tiktoken_encoding.encode(sentence))
            except Exception as e:
                logger.warning(f"tiktoken encoding failed for sentence, falling back to space split. Error: {e}")
                sentence_token_count = len(sentence.split())
        else:
            sentence_token_count = len(sentence.split())

        # If one sentence alone exceeds l_max, finalize the current chunk if non-empty,
        # then store this sentence as its own chunk.
        if sentence_token_count >= l_max_tokens:
            # Dump the current chunk
            if len(current_chunk) > 0:
                chunk_str = " ".join(current_chunk)
                chunks.append(SingleHopChunk(chunk_id=f"{doc_id}_{chunk_index}", chunk_text=chunk_str))
                chunk_index += 1
                current_chunk = []
                current_len = 0
            # Store the sentence alone
            chunks.append(SingleHopChunk(chunk_id=f"{doc_id}_{chunk_index}", chunk_text=sentence))
            chunk_index += 1
            continue

        # Otherwise, add this sentence to the current chunk
        current_chunk.append(sentence)
        current_len += sentence_token_count

        # If we exceed l_max, close the current chunk and start a new one
        if current_len >= l_max_tokens:
            chunk_str = " ".join(current_chunk)
            chunks.append(SingleHopChunk(chunk_id=f"{doc_id}_{chunk_index}", chunk_text=chunk_str))
            chunk_index += 1
            current_chunk = []
            current_len = 0
            continue

        # If we have at least l_min tokens and the next sentence similarity is below threshold, break here
        if (current_len >= l_min_tokens) and (i < len(sentences) - 1):
            if similarities[i] < tau:
                chunk_str = " ".join(current_chunk)
                chunks.append(SingleHopChunk(chunk_id=f"{doc_id}_{chunk_index}", chunk_text=chunk_str))
                chunk_index += 1
                current_chunk = []
                current_len = 0

    # Any leftover
    if len(current_chunk) > 0:
        chunk_str = " ".join(current_chunk)
        chunks.append(SingleHopChunk(chunk_id=f"{doc_id}_{chunk_index}", chunk_text=chunk_str))

    return chunks


# Define terminators for the improved fast chunking
_FAST_CHUNKING_TERMINATORS = [
    "。", "？", "！", "；", "……", "…", "》", "】", "）",
    ".", "?", "!", ";", "…", '"', ")", "]", "}" # Added "." and standard quote
]

def _chunk_document_fast_fallback(
    sentences: list[str],
    l_max_tokens: int,
    doc_id: str,
    use_tiktoken: bool,
) -> list[SingleHopChunk]:
    """
    Creates chunks based purely on a maximum token length. Each sentence is added
    to the current chunk if it does not exceed l_max_tokens; otherwise, a new chunk
    is started.

    Args:
        sentences (list[str]): The list of sentences for a single document.
        l_max_tokens (int): Maximum tokens per chunk.
        doc_id (str): Unique identifier for the document.
        use_tiktoken (bool): If True, use tiktoken for counting. Otherwise, split by space.

    Returns:
        list[SingleHopChunk]: A list of SingleHopChunk objects.
    """
    chunks: list[SingleHopChunk] = []
    current_chunk: list[str] = []
    current_len: int = 0
    chunk_index: int = 0

    for sentence in sentences:
        # Calculate token count based on availability
        if use_tiktoken and _tiktoken_encoding:
            try:
                sentence_token_count = len(_tiktoken_encoding.encode(sentence))
            except Exception as e:
                logger.warning(f"tiktoken encoding failed for sentence, falling back to space split. Error: {e}")
                sentence_token_count = len(sentence.split())
        else:
            sentence_token_count = len(sentence.split())

        # If adding this sentence would exceed l_max_tokens, finalize current chunk
        if current_len + sentence_token_count > l_max_tokens and current_chunk:
            if current_chunk:
                chunk_str = " ".join(current_chunk)
                chunks.append(SingleHopChunk(chunk_id=f"{doc_id}_{chunk_index}", chunk_text=chunk_str))
                chunk_index += 1

            # Start a new chunk with the current sentence
            current_chunk = [sentence]
            current_len = sentence_token_count
        else:
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_len += sentence_token_count

    # Any leftover chunk
    if current_chunk:
        chunk_str = " ".join(current_chunk)
        chunks.append(SingleHopChunk(chunk_id=f"{doc_id}_{chunk_index}", chunk_text=chunk_str))

    return chunks


def _chunk_document_fast_tiktoken(
    text:str,
    l_min_tokens:int,
    l_max_tokens:int,
    doc_id: str,
    tokenizer: tiktoken.Encoding
)->list[SingleHopChunk]:
    """
    Splits text into chunks using terminators and token length constraints (min/max).
    Requires tiktoken tokenizer.

    Args:
        text (str): The input document text.
        l_min_tokens (int): Minimum token length for a chunk.
        l_max_tokens (int): Maximum token length for a chunk.
        doc_id (str): Document ID for generating chunk IDs.
        tokenizer (tiktoken.Encoding): The tiktoken tokenizer instance.

    Returns:
        list[SingleHopChunk]: A list of single-hop chunks.
    """
    tokens = tokenizer.encode(text, allowed_special=set())
    chunks_result: list[SingleHopChunk] = []
    chunk_index = 0

    # Split tokens into sentences based on terminators
    sentences_tokens: list[list[int]] = []
    last_idx = 0

    for i, token in enumerate(tokens):
        try:
            token_text = tokenizer.decode([token])
            # Check if the decoded token is one of the terminators
            if token_text in _FAST_CHUNKING_TERMINATORS:
                if i + 1 > last_idx:  # Ensure sentence is not empty
                    sentences_tokens.append(tokens[last_idx:i+1])
                    last_idx = i + 1
        except Exception as e:
            # Handle potential decoding errors if necessary, though unlikely with allowed_special=set()
            logger.trace(f"Could not decode token {token} at index {i}. Skipping terminator check for this token. Error: {e}")
            continue # Skip terminator check if decoding fails

    # Handle the last sentence if it doesn't end with a terminator
    if last_idx < len(tokens):
        sentences_tokens.append(tokens[last_idx:])

    if not sentences_tokens: # Handle case where no sentences could be formed
        if tokens: # If there are tokens, treat the whole text as one chunk (respecting max_length)
            if len(tokens) <= l_max_tokens:
                 chunks_result.append(SingleHopChunk(chunk_id=f"{doc_id}_{chunk_index}", chunk_text=tokenizer.decode(tokens)))
            else: # If even the whole text is too long, truncate it (or split further if needed)
                 logger.warning(f"Document {doc_id} could not be split into sentences and exceeds max_length. Truncating.")
                 chunk_text = tokenizer.decode(tokens[:l_max_tokens])
                 chunks_result.append(SingleHopChunk(chunk_id=f"{doc_id}_{chunk_index}", chunk_text=chunk_text))
        return chunks_result # Return empty or the single chunk


    # Merge sentences into chunks using a sliding window approach
    current_chunk_tokens: list[int] = []

    for sentence_tokens in sentences_tokens:
        # If current chunk + new sentence doesn't exceed max_length, add it
        if len(current_chunk_tokens) + len(sentence_tokens) <= l_max_tokens:
            current_chunk_tokens.extend(sentence_tokens)
        else:
            # If current chunk meets min_length, save it
            if len(current_chunk_tokens) >= l_min_tokens:
                chunk_text = tokenizer.decode(current_chunk_tokens)
                chunks_result.append(SingleHopChunk(chunk_id=f"{doc_id}_{chunk_index}", chunk_text=chunk_text))
                chunk_index += 1
                # Start new chunk with the current sentence
                current_chunk_tokens = sentence_tokens
            else:
                # Current chunk is too short, but adding the new sentence makes it too long.
                # Fill the current chunk up to max_length from the new sentence.
                space_left = l_max_tokens - len(current_chunk_tokens)
                if space_left > 0 : # Ensure we actually add something
                    current_chunk_tokens.extend(sentence_tokens[:space_left])

                chunk_text = tokenizer.decode(current_chunk_tokens)
                chunks_result.append(SingleHopChunk(chunk_id=f"{doc_id}_{chunk_index}", chunk_text=chunk_text))
                chunk_index += 1

                # Start new chunk with the remainder of the sentence
                current_chunk_tokens = sentence_tokens[space_left:]

            # Special case: If the new sentence *itself* is longer than max_length after the split
            while len(current_chunk_tokens) > l_max_tokens:
                 chunk_text = tokenizer.decode(current_chunk_tokens[:l_max_tokens])
                 chunks_result.append(SingleHopChunk(chunk_id=f"{doc_id}_{chunk_index}", chunk_text=chunk_text))
                 chunk_index += 1
                 current_chunk_tokens = current_chunk_tokens[l_max_tokens:]


    # Handle the last remaining chunk
    if current_chunk_tokens: # Check if there are remaining tokens
        if len(current_chunk_tokens) >= l_min_tokens:
            chunk_text = tokenizer.decode(current_chunk_tokens)
            chunks_result.append(SingleHopChunk(chunk_id=f"{doc_id}_{chunk_index}", chunk_text=chunk_text))
        elif chunks_result:  # If the last chunk is too short but not empty, try merging with the previous one
            # Get the tokens of the last saved chunk
            # Re-encoding the last chunk text might differ slightly from original tokens, be cautious
            # A safer approach might be to keep track of the *tokens* of the last chunk.
            # For simplicity now, we decode and check length, then potentially merge text.
            last_chunk_text = chunks_result[-1].chunk_text
            last_chunk_tokens = tokenizer.encode(last_chunk_text, allowed_special=set()) # Re-encode might not be perfect

            if len(last_chunk_tokens) + len(current_chunk_tokens) <= l_max_tokens:
                 # Merge tokens and update the last chunk's text
                 merged_tokens = last_chunk_tokens + current_chunk_tokens
                 chunks_result[-1].chunk_text = tokenizer.decode(merged_tokens)
            else:
                 # Cannot merge, and it's too short. Discard or log as needed.
                 logger.debug(f"Discarding final short chunk for doc {doc_id} as it couldn't be merged.")
                 # Alternatively, keep it if requirements allow very short final chunks:
                 # chunk_text = tokenizer.decode(current_chunk_tokens)
                 # chunks_result.append(SingleHopChunk(chunk_id=f"{doc_id}_{chunk_index}", chunk_text=chunk_text))

        # If it's short and there are no previous chunks (meaning the entire doc resulted in one short chunk)
        elif len(current_chunk_tokens) > 0: # Keep it if it's the only content
             chunk_text = tokenizer.decode(current_chunk_tokens)
             chunks_result.append(SingleHopChunk(chunk_id=f"{doc_id}_{chunk_index}", chunk_text=chunk_text))


    return chunks_result


def _multihop_chunking(
    single_hop_chunks: list[SingleHopChunk],
    h_min: int,
    h_max: int,
    num_multihops_factor: int,
) -> list[MultiHopChunk]:
    """
    Creates multi-hop chunks by generating all valid combinations of single-hop chunks
    (from size h_min to h_max), then shuffling and picking the desired number. This
    ensures no repeated multi-hop chunk grouping is created.

    The total multi-hop chunks to select is determined by:
        num_multihops = max(1, total_single_hops // num_multihops_factor).

    If the number of possible unique combinations is less than or equal to num_multihops,
    we return all. Otherwise, we select a random sample of size num_multihops from those
    unique combinations.

    Args:
        single_hop_chunks (list[SingleHopChunk]): list of single-hop chunk objects.
        h_min (int): Minimum number of chunks to combine.
        h_max (int): Maximum number of chunks to combine.
        num_multihops_factor (int): Determines how many multi-hop chunks to generate,
                                    typically a fraction of the total single-hop chunks.

    Returns:
        list[MultiHopChunk]: The resulting multi-hop chunk objects.
    """
    if single_hop_chunks is None or len(single_hop_chunks) == 0:
        return []

    total_single_hops = len(single_hop_chunks)
    # This is our target count for how many multi-hop combos we want to keep
    num_multihops = max(1, total_single_hops // num_multihops_factor)

    # Build a list of ALL possible multi-hop combinations from h_min to h_max
    all_combos: list[MultiHopChunk] = []
    for size in range(h_min, h_max + 1):
        if size > total_single_hops:
            break
        for combo_indices in itertools.combinations(range(total_single_hops), size):
            chosen_chunks = [single_hop_chunks[idx] for idx in combo_indices]
            group_obj = MultiHopChunk(
                chunk_ids=[c.chunk_id for c in chosen_chunks],
                chunks_text=[c.chunk_text for c in chosen_chunks],
            )
            all_combos.append(group_obj)

    random.shuffle(all_combos)
    if len(all_combos) <= num_multihops:
        return all_combos
    else:
        return all_combos[:num_multihops]


def _compute_info_density_metrics(
    chunks: list[SingleHopChunk],
    local_perplexity_metric: Optional[Any],
    local_use_textstat: bool,
    use_tiktoken: bool,
) -> list[ChunkInfoMetrics]:
    """
    Computes optional statistics for each chunk, including token count, perplexity,
    readability (flesch, gunning fog), and basic lexical diversity metrics.

    Args:
        chunks (list[SingleHopChunk]): The list of single-hop chunk objects.
        local_perplexity_metric (Optional[Any]): If provided, used to compute
                                                 perplexity (from evaluate.load("perplexity")).
        local_use_textstat (bool): If True, compute text readability metrics using textstat.
        use_tiktoken (bool): If True, use tiktoken for counting token_count. Otherwise, split by space.

    Returns:
        list[ChunkInfoMetrics]: One object per chunk with fields like:
          - token_count
          - unique_token_ratio
          - bigram_diversity
          - perplexity
          - avg_token_length
          - flesch_reading_ease
          - gunning_fog
    """
    results: list[ChunkInfoMetrics] = []

    for chunk in chunks:
        chunk_text: str = chunk.chunk_text
        tokens = chunk_text.strip().split()
        word_token_count = len(tokens)

        # Calculate token count using tiktoken if available, else use word count
        actual_token_count = 0
        if use_tiktoken and _tiktoken_encoding:
            try:
                actual_token_count = len(_tiktoken_encoding.encode(chunk_text))
            except Exception as e:
                logger.warning(f"tiktoken encoding failed for chunk, falling back to space split count. Error: {e}")
                actual_token_count = word_token_count
        else:
            actual_token_count = word_token_count

        # Compute metrics step by step
        unique_token_ratio = 0.0
        if word_token_count > 0:
            unique_toks = len({t.lower() for t in tokens})
            unique_token_ratio = float(unique_toks / word_token_count)

        # Bigram diversity (word-based)
        bigram_diversity = 0.0
        if word_token_count > 1:
            bigrams = []
            for i in range(word_token_count - 1):
                bigrams.append((tokens[i].lower(), tokens[i + 1].lower()))
            unique_bigrams = len(set(bigrams))
            bigram_diversity = float(unique_bigrams / len(bigrams))

        # Perplexity (uses original text)
        ppl_score: float = 0.0
        if local_perplexity_metric is not None and chunk_text.strip():
            try:
                result = local_perplexity_metric.compute(data=[chunk_text], batch_size=1)
                ppl_score = result.get("mean_perplexity", 0.0)
            except Exception as e:
                logger.warning(f"Could not compute perplexity for chunk. Error: {e}")
                ppl_score = 0.0

        # Average token length (word-based)
        avg_token_length = 0.0
        if word_token_count > 0:
            avg_len = sum(len(t) for t in tokens) / word_token_count
            avg_token_length = float(avg_len)

        # Readability (uses original text)
        flesch_reading_ease = 0.0
        gunning_fog = 0.0
        if local_use_textstat is True and chunk_text.strip():
            try:
                flesch_reading_ease = float(textstat.flesch_reading_ease(chunk_text))
                gunning_fog = float(textstat.gunning_fog(chunk_text))
            except Exception as e:
                logger.warning(f"Textstat error: {e}")

        results.append(
            ChunkInfoMetrics(
                token_count=float(actual_token_count),
                unique_token_ratio=unique_token_ratio,
                bigram_diversity=bigram_diversity,
                perplexity=ppl_score,
                avg_token_length=avg_token_length,
                flesch_reading_ease=flesch_reading_ease,
                gunning_fog=gunning_fog,
            )
        )

    return results


def _plot_aggregated_similarities(all_similarities: list[list[float]]) -> None:
    """
    Plots the average cosine similarity for each sentence-pair position across
    all documents, with shaded regions representing one standard deviation.

    Args:
        all_similarities (list[list[float]]): A list of lists, where each
            sub-list is the array of consecutive sentence similarities for
            a particular document.
    """
    if all_similarities is None or len(all_similarities) == 0:
        logger.debug("No similarities to plot. Skipping aggregated similarity plot.")
        return

    # Check if matplotlib is available before trying to plot
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Matplotlib not found. Skipping similarity plot generation.")
        return

    plt.figure(figsize=(10, 6))
    max_len = max(len(sims) for sims in all_similarities)

    avg_sim: list[float] = []
    std_sim: list[float] = []
    counts: list[int] = []

    for position in range(max_len):
        vals = [s[position] for s in all_similarities if position < len(s)]
        if vals:
            mean_val = sum(vals) / len(vals)
            variance = sum((v - mean_val) ** 2 for v in vals) / len(vals)
            stddev_val = variance**0.5

            avg_sim.append(mean_val)
            std_sim.append(stddev_val)
            counts.append(len(vals))
        else:
            break

    # X-axis positions
    x_positions = list(range(len(avg_sim)))
    plt.plot(x_positions, avg_sim, "b-", label="Avg Similarity")

    # Create confidence interval region
    lower_bound = [max(0, a - s) for a, s in zip(avg_sim, std_sim)]
    upper_bound = [min(1, a + s) for a, s in zip(avg_sim, std_sim)]
    plt.fill_between(x_positions, lower_bound, upper_bound, alpha=0.3, color="blue")

    # Plot data points with size reflecting how many docs contributed
    max_count = max(counts) if counts else 1
    sizes = [30.0 * (c / max_count) for c in counts]
    plt.scatter(x_positions, avg_sim, s=sizes, alpha=0.5, color="navy")

    plt.title("Average Consecutive Sentence Similarity Across Documents")
    plt.xlabel("Sentence Pair Index")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)
    plot_path: str = os.path.join("plots", "aggregated_similarities.png")
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")  # Changed dpi to 300
    plt.close()
    logger.info(f"Saved aggregated similarity plot at '{plot_path}'.")


# Make sure main guard exists if this file is runnable directly (optional but good practice)
if __name__ == "__main__":
    # Example configuration for testing (replace with actual loading if needed)
    test_config = {
        "pipeline": {
            "chunking": {
                "run": True,
                "chunking_configuration": {
                    "chunking_mode": "fast_chunking"  # or "semantic_chunking" if deps installed
                },
                # Add other necessary config keys like dataset paths etc.
            }
        },
        "settings": {"debug": True},
        # Add dataset config, model roles etc.
    }
    # Basic logger setup for standalone execution
    logger.add("logs/chunking_standalone.log", rotation="10 MB")
    logger.info("Running chunking module standalone (example)...")
    # Note: You'd need a valid dataset configuration for run() to work fully.
    # run(test_config)
    logger.info("Standalone example finished.")
