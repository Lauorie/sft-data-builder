# ingestion.py

"""
Author: @sumukshashidhar

This module implements the "ingestion" stage of the YourBench pipeline.

Purpose:
    The ingestion stage reads source documents from a user-specified directory,
    converts each document into markdown using the external 'magic-pdf' tool, and
    saves the converted outputs in the specified output directory. This normalized
    markdown output sets the foundation for subsequent pipeline steps.

Usage:
    from yourbench.pipeline import ingestion
    ingestion.run(config)

Configuration Requirements (in `config["pipeline"]["ingestion"]`):
    {
      "run": bool,  # Whether to enable the ingestion stage
      "source_documents_dir": str,  # Directory containing raw source documents
      "output_dir": str,           # Directory where converted .md files are saved
    }

Stage-Specific Logging:
    All major ingestion activity is logged to "logs/ingestion.log".
"""

import os
import glob
import subprocess
import shutil
from typing import Any, Optional
from dataclasses import field, dataclass

from loguru import logger

# Removed Hugging Face and YourBench model imports as they are no longer used
# from huggingface_hub import InferenceClient
# from yourbench.utils.inference_engine import Model as ModelConfig


@dataclass
class IngestionConfig:
    """Configuration for the ingestion stage of the pipeline."""

    run: bool = False
    source_documents_dir: Optional[str] = None
    output_dir: Optional[str] = None


# Removed unused dataclasses ModelRoles and PipelineConfig
# @dataclass
# class ModelRoles:
#     """Configuration for model roles in the pipeline."""
#
#     ingestion: list[str] = field(default_factory=list)
#
#
# @dataclass
# class PipelineConfig:
#     """Main configuration for the pipeline."""
#
#     pipeline: dict[str, Any] = field(default_factory=dict)
#     model_roles: ModelRoles = field(default_factory=ModelRoles)
#     model_list: list[ModelConfig] = field(default_factory=list)


def _extract_ingestion_config(config: dict[str, Any]) -> IngestionConfig:
    """
    Extract ingestion configuration from the main config dictionary.

    Args:
        config (dict[str, Any]): The complete configuration dictionary.

    Returns:
        IngestionConfig: A typed configuration object for ingestion.
    """
    if not isinstance(config.get("pipeline", {}).get("ingestion", {}), dict):
        return IngestionConfig()

    stage_config = config.get("pipeline", {}).get("ingestion", {})
    return IngestionConfig(
        run=stage_config.get("run", False),
        source_documents_dir=stage_config.get("source_documents_dir"),
        output_dir=stage_config.get("output_dir"),
    )


# Removed unused helper functions _extract_model_roles and _extract_model_list
# def _extract_model_roles(config: dict[str, Any]) -> ModelRoles:
#     """
#     Extract model roles configuration from the main config dictionary.
#
#     Args:
#         config (dict[str, Any]): The complete configuration dictionary.
#
#     Returns:
#         ModelRoles: A typed configuration object for model roles.
#     """
#     model_roles_dict = config.get("model_roles", {})
#     return ModelRoles(ingestion=model_roles_dict.get("ingestion", []))
#
#
# def _extract_model_list(config: dict[str, Any]) -> list[ModelConfig]:
#     """
#     Extract model list configuration from the main config dictionary.
#
#     Args:
#         config (dict[str, Any]): The complete configuration dictionary.
#
#     Returns:
#         list[ModelConfig]: A list of typed model configurations.
#     """
#     model_list_dicts = config.get("model_list", [])
#     result = []
#
#     for model_dict in model_list_dicts:
#         model_config = ModelConfig(
#             model_name=model_dict.get("model_name"),
#             base_url=model_dict.get("base_url"),
#             api_key=model_dict.get("api_key"),
#             provider=model_dict.get("provider"),
#         )
#         result.append(model_config)
#
#     return result


def run(config: dict[str, Any]) -> None:
    """
    Execute the ingestion stage of the pipeline using magic-pdf.

    This function checks whether the ingestion stage is enabled in the pipeline
    configuration. If enabled, it performs the following actions:

    1. Runs the 'magic-pdf' command-line tool to convert all documents
       in the `source_documents_dir` to Markdown.
    2. Saves the resulting .md outputs to a temporary structure within `output_dir`.
    3. Moves the converted .md files from the temporary structure to the root of `output_dir`.
    4. Cleans up the temporary directories created by magic-pdf.

    Args:
        config (dict[str, Any]): A configuration dictionary containing the
            'pipeline.ingestion' section with keys:
            - "run" (bool): Whether to run ingestion.
            - "source_documents_dir" (str): Directory containing source documents.
            - "output_dir" (str): Directory where final .md files will be saved.

    Returns:
        None

    Logs:
        Writes detailed logs to logs/ingestion.log describing each step taken
        and any errors encountered during the process.
    """
    # Extract typed configurations from the dictionary
    ingestion_config = _extract_ingestion_config(config)

    # Check if ingestion is enabled
    if not ingestion_config.run:
        logger.info("Ingestion stage is disabled. No action will be taken.")
        return

    # Check required directories
    if not ingestion_config.source_documents_dir or not ingestion_config.output_dir:
        logger.error("Missing 'source_documents_dir' or 'output_dir' in ingestion config. Cannot proceed.")
        return

    source_dir = ingestion_config.source_documents_dir
    output_dir = ingestion_config.output_dir

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    logger.debug("Prepared output directory: {}", output_dir)

    logger.info(
        "Ingestion stage: Converting documents from '{}' to '{}' using magic-pdf...",
        source_dir,
        output_dir,
    )

    # --- Execute magic-pdf command ---
    # The command structure: magic-pdf -p <input_dir> -o <output_dir> -m auto
    magic_pdf_command = [
        "magic-pdf",
        "-p", source_dir,
        "-o", output_dir,
        "-m", "auto"
    ]
    logger.info(f"Running command: {' '.join(magic_pdf_command)}")

    try:
        # Using utf-8 encoding explicitly for cross-platform compatibility
        result = subprocess.run(magic_pdf_command, check=True, capture_output=True, text=True, encoding='utf-8')
        logger.info("magic-pdf command completed successfully.")
        logger.debug(f"magic-pdf stdout:\n{result.stdout}")
        if result.stderr:
            # magic-pdf might output progress or info to stderr
            logger.info(f"magic-pdf stderr:\n{result.stderr}")
    except FileNotFoundError:
         logger.error("Error: 'magic-pdf' command not found. Make sure it is installed and in your PATH.")
         return
    except subprocess.CalledProcessError as e:
        logger.error(f"magic-pdf command failed with return code {e.returncode}.")
        logger.error(f"Command: {' '.join(e.cmd)}")
        # Decode stderr/stdout safely in case of encoding issues
        stderr_output = e.stderr if e.stderr else ""
        stdout_output = e.stdout if e.stdout else ""
        logger.error(f"Stderr:\n{stderr_output}")
        logger.error(f"Stdout:\n{stdout_output}")
        logger.error("Ingestion failed due to magic-pdf error.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred while running magic-pdf: {e}")
        return

    # --- Move and Clean Up Files ---
    logger.info("Moving converted markdown files to the root output directory and cleaning up...")
    moved_files_count = 0
    cleaned_dirs_count = 0

    # Iterate through items potentially created by magic-pdf in the output directory
    # The structure is expected to be output_dir/original_filename_no_ext/auto/original_filename_no_ext.md
    try:
        # List items in the output directory AFTER magic-pdf has run
        items_in_output_dir = os.listdir(output_dir)
        logger.debug(f"Items found in output directory after magic-pdf run: {items_in_output_dir}")

        for item_name in items_in_output_dir:
            item_path = os.path.join(output_dir, item_name)

            # Check if it's a directory (magic-pdf creates subdirs named after input files)
            if os.path.isdir(item_path):
                # Define the expected path components based on magic-pdf's structure
                auto_dir = os.path.join(item_path, "auto")
                # The markdown filename inside 'auto' should match the directory name 'item_name'
                expected_md_filename = f"{item_name}.md"
                source_md_path = os.path.join(auto_dir, expected_md_filename)

                # Verify the expected structure exists before attempting to move/delete
                if os.path.isdir(auto_dir) and os.path.isfile(source_md_path):
                    target_md_path = os.path.join(output_dir, expected_md_filename)

                    try:
                        # Move the .md file to the root output directory
                        shutil.move(source_md_path, target_md_path)
                        logger.debug(f"Moved '{source_md_path}' to '{target_md_path}'")
                        moved_files_count += 1

                        # Attempt to remove the intermediate directory structure (item_path)
                        # This includes the 'auto' subdirectory and potentially other files magic-pdf left
                        shutil.rmtree(item_path)
                        logger.debug(f"Removed intermediate directory: '{item_path}'")
                        cleaned_dirs_count += 1
                    except OSError as move_err:
                         # Catch potential errors during move or rmtree (e.g., permissions, file locks)
                        logger.error(f"Error processing '{item_path}': Could not move file or remove directory. Error: {move_err}")
                    except Exception as generic_err:
                        logger.error(f"Unexpected error processing '{item_path}': {generic_err}")

                else:
                    # Log if the expected internal structure wasn't found within a directory.
                    # This might happen if magic-pdf failed for a specific file or changed its output format.
                    logger.warning(
                        f"Expected magic-pdf output structure "
                        f"('{os.path.join('auto', expected_md_filename)}') "
                        f"not found within directory '{item_path}'. Skipping cleanup for this item."
                     )
            # else: It's a file directly in output_dir, might be unrelated or an unexpected output. Ignore for cleanup.


    except FileNotFoundError:
        logger.error(f"Output directory '{output_dir}' not found after magic-pdf run. Cannot proceed with cleanup.")
    except PermissionError:
        logger.error(f"Permission denied while accessing or modifying files/directories in '{output_dir}'.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during file moving and cleanup: {e}")
        # Depending on severity, might want to re-raise or handle differently

    if moved_files_count > 0 or cleaned_dirs_count > 0 :
        logger.success(
            f"Ingestion stage complete using magic-pdf. Moved {moved_files_count} markdown files. "
            f"Cleaned up {cleaned_dirs_count} intermediate directories. Output is in '{output_dir}'."
        )
    else:
         logger.warning(
             f"Ingestion stage finished, but no files were moved or cleaned up. "
             f"Check if magic-pdf produced the expected output structure in '{output_dir}' "
             f"and if the source directory '{source_dir}' contained processable files."
         )
