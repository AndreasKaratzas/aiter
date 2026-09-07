# SPDX-License-Identifier: MIT
"""Load benchmark shapes without importing a GPU runtime.

The bundled catalog is a package resource. Explicit user paths are resolved by
normal filesystem rules relative to the caller's working directory.
"""

import json
from importlib.resources import files
from pathlib import Path


def load_model_configs(path=None):
    source = (
        files("benchmarks.models").joinpath("model_configs.json")
        if path is None
        else Path(path)
    )
    return json.loads(source.read_text())


def get_model_configs(config_path=None, models="llama3,mixtral_7B"):
    """
    Load model names from the configuration file.

    Args:
        config_path (str): User-provided path to the configuration JSON file.
        models: List of model names to retrieve, with pattern <modelfamily_modelsize>. If modelfamily specified only, retrieves all the modelsizes.

    Returns:
        dict: A dictionary of available models and their configurations for the specified families.
    """
    configs = load_model_configs(config_path)

    # Extract models and their configurations for the specified families
    filtered_configs = {}

    if models == "all":
        models = [model for model in configs]
    else:
        models = models.replace(" ", "").split(",")

    for model in models:
        delimiter = "_" if "_" in model else "-"
        model_specs = model.split(delimiter)
        model_family = model_specs[0]

        if model_family in configs:
            model_size = model_specs[1] if len(model_specs) > 1 else None
            # Check if model filtering is required
            if model_size is None:  # Include all models in the family
                # Include all models in the family
                for model_size, model_configs in configs[model_family].items():
                    filtered_configs[f"{model_family}-{model_size}"] = model_configs
            else:
                if model_size in configs[model_family]:
                    filtered_configs[f"{model_family}-{model_size}"] = configs[
                        model_family
                    ][model_size]

    if not filtered_configs:
        print(f"Warning: No models selected with the provided model names: {models}")

    return filtered_configs


def get_available_models(config_file=None, filter=None):
    """
    Load model names from the configuration file.

    Args:
        config_file (str): Path to the configuration JSON file.

    Returns:
        list: A list of available model configs.
    """
    configs = load_model_configs(config_file)

    models = [
        f"{family}-{model}"
        for family in configs
        for model in configs[family]
        if filter is None or filter in f"{family}-{model}"
    ]

    return models
