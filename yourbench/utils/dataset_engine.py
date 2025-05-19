import os
from typing import Any, Dict, Optional

from loguru import logger

from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets, load_from_disk


class ConfigurationError(Exception):
    """Exception raised for errors in the configuration."""

    pass


def _get_full_dataset_repo_name(config: Dict[str, Any]) -> str:
    try:
        if "hf_configuration" not in config:
            error_msg = "Missing 'hf_configuration' in config"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)

        hf_config = config["hf_configuration"]
        if "hf_dataset_name" not in hf_config:
            error_msg = "Missing 'hf_dataset_name' in hf_configuration"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)

        dataset_name = hf_config["hf_dataset_name"]
        if "/" not in dataset_name:
            dataset_name = f"{hf_config['hf_organization']}/{dataset_name}"

        return dataset_name
    except ConfigurationError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise e


def custom_load_dataset(config: Dict[str, Any], subset: Optional[str] = None) -> Dataset:
    """
    Load a dataset subset strictly from the local path specified in the config.
    Raises FileNotFoundError if the required local dataset subset is not found.
    """
    local_dataset_dir = config.get("local_dataset_dir")

    if not local_dataset_dir:
        error_msg = "Configuration error: 'local_dataset_dir' is not specified in the config. Cannot load dataset."
        logger.error(error_msg)
        # Use ConfigurationError or ValueError might be appropriate here too
        raise ConfigurationError(error_msg)

    # Determine the expected path for the subset
    if subset:
        local_subset_path = os.path.join(local_dataset_dir, subset)
    else:
        # If no subset name, perhaps loading the base dir directly? Adjust if needed.
        # Assuming saving puts subsets in subdirs, this case might be less common.
        local_subset_path = local_dataset_dir
        logger.warning(f"No subset specified, attempting to load directly from '{local_subset_path}'. This might fail if data is saved in subset subdirectories.")

    # Check if the path exists
    if not os.path.exists(local_subset_path):
        error_msg = f"Required local dataset subset not found at path: '{local_subset_path}'. Please ensure the previous pipeline stage successfully generated this data."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Try loading from the existing local path
    logger.info(f"Found local dataset at '{local_subset_path}'. Loading from disk.")
    try:
        loaded_data = load_from_disk(local_subset_path)

        # Handle DatasetDict (standard save_to_disk format for subsets)
        if isinstance(loaded_data, DatasetDict):
            if subset and subset in loaded_data:
                 logger.success(f"Successfully loaded subset '{subset}' from '{local_subset_path}'.")
                 return loaded_data[subset]
            elif not subset and "train" in loaded_data: # Fallback for base dir load?
                 logger.success(f"Successfully loaded default 'train' split from '{local_subset_path}'.")
                 return loaded_data["train"]
            else:
                 # If DatasetDict loaded but the expected key isn't there
                 error_msg = f"Loaded DatasetDict from '{local_subset_path}' but could not find the expected key '{subset or 'train'}'. Check saved data structure."
                 logger.error(error_msg)
                 # Treat this as data not found/corrupt
                 raise FileNotFoundError(error_msg)

        # Handle direct Dataset loading (less common if saved via DatasetDict)
        elif isinstance(loaded_data, Dataset):
             logger.success(f"Successfully loaded Dataset directly from '{local_subset_path}'.")
             return loaded_data
        else:
            # Loaded something unexpected
            error_msg = f"Loaded data from '{local_subset_path}' is not a Dataset or DatasetDict. Type: {type(loaded_data)}"
            logger.error(error_msg)
            raise TypeError(error_msg) # Or maybe FileNotFoundError/ValueError

    except Exception as e:
        # Catch exceptions during load_from_disk or subsequent processing
        error_msg = f"Failed to load dataset from local path '{local_subset_path}': {e}"
        logger.error(error_msg)
        # Re-raise the original error or a more specific one like FileNotFoundError
        # Using FileNotFoundError implies the data isn't usable/present correctly.
        raise FileNotFoundError(f"Error loading dataset from '{local_subset_path}'. Original error: {e}")


def custom_save_dataset(
    dataset: Dataset,
    config: Dict[str, Any],
    subset: Optional[str] = None,
    save_local: bool = True,
    push_to_hub: bool = True,
) -> None:
    """
    Save a dataset subset locally (in Arrow format and optionally JSON Lines format)
    and potentially push it to Hugging Face Hub.
    """

    dataset_repo_name = _get_full_dataset_repo_name(config)

    local_dataset_dir = config.get("local_dataset_dir", None)
    if local_dataset_dir and save_local:
        logger.info(f"Saving dataset locally to base directory: '{local_dataset_dir}'")

        # Determine paths for Arrow and JSON formats
        if subset:
            # Arrow format will be saved inside a subdirectory named after the subset
            arrow_save_path = os.path.join(local_dataset_dir, subset)
            # JSON format will be saved as a file parallel to the Arrow directory
            json_save_path = os.path.join(local_dataset_dir, f"{subset}.json")
            # Prepare DatasetDict for save_to_disk
            local_dataset_to_save = DatasetDict({subset: dataset})
        else:
            # If no subset, save Arrow directly in the base directory
            arrow_save_path = local_dataset_dir
            # Default name for JSON file if no subset is specified
            json_save_path = os.path.join(local_dataset_dir, "data.json")
            local_dataset_to_save = dataset # Save the dataset directly

        # Ensure the directory for Arrow format exists
        os.makedirs(arrow_save_path, exist_ok=True)
        # Save in Arrow format
        local_dataset_to_save.save_to_disk(arrow_save_path)
        logger.success(f"Dataset successfully saved locally to: '{arrow_save_path}' (Arrow format)")

        # --- BEGIN MODIFIED JSON SAVING ---
        # Change file extension
        if json_save_path.endswith(".jsonl"):
             json_save_path = json_save_path[:-1] # Change .jsonl to .json

        try:
            os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
            # Use lines=False, add indent=4, keep force_ascii=False, orient='records' is fine
            dataset.to_json(json_save_path, lines=False, orient="records", indent=4, force_ascii=False)
            logger.success(f"Dataset also saved locally to: '{json_save_path}' (Pretty JSON format)")
        except Exception as e:
            logger.error(f"Failed to save dataset to pretty JSON format at '{json_save_path}': {e}")
        # --- END MODIFIED JSON SAVING ---

    # TODO: add this back in
    # if config["hf_configuration"].get("concat_if_exist", False):
    #     existing_dataset = custom_load_dataset(config=config, subset=subset)
    #     dataset = concatenate_datasets([existing_dataset, dataset])
    #     logger.info("Concatenated dataset with an existing one")

    # if subset:
    #     config_name = subset
    # else:
    #     config_name = "default"

    # if push_to_hub:
    #     logger.info(f"Pushing dataset to HuggingFace Hub with repo_id='{dataset_repo_name}'")
    #     dataset.push_to_hub(
    #         repo_id=dataset_repo_name,
    #         private=config["hf_configuration"].get("private", True),
    #         config_name=config_name,
    #     )
    #     logger.success(f"Dataset successfully pushed to HuggingFace Hub with repo_id='{dataset_repo_name}'")
