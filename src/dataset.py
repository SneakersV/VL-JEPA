import json
from io import BytesIO

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm


DEFAULT_REPO_ID = "Dangindev/viet-cultural-vqa"

SPLIT_TO_FILENAME = {
    "train": "splits/train_data.json",
    "validation": "splits/val_data.json",
    "val": "splits/val_data.json",
    "test": "splits/test_data.json",
}


def normalize_path(path):
    return path.replace("\\", "/")


def resolve_image_path(image_path):
    if not image_path:
        return None

    image_path = normalize_path(image_path)

    if image_path.startswith("/data/"):
        image_path = image_path[len("/data/"):]
    elif image_path.startswith("data/"):
        image_path = image_path[len("data/"):]

    return image_path


def _resolve_split_filename(split):
    split_key = split.lower()

    if split_key not in SPLIT_TO_FILENAME:
        valid_splits = ", ".join(sorted(SPLIT_TO_FILENAME))
        raise ValueError(f"Unknown split '{split}'. Expected one of: {valid_splits}")

    return SPLIT_TO_FILENAME[split_key]


def _download_image(image_url, timeout=30):
    response = requests.get(image_url, timeout=timeout)
    response.raise_for_status()

    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    return np.array(pil_image)


def load_viet_cultural_split(
    split="train",
    repo_id=DEFAULT_REPO_ID,
    n_samples=None,
    request_timeout=30,
):
    """
    Load one Vietnamese Cultural VQA split from Hugging Face.

    This uses the repo JSON split files and manually downloads image files so
    the returned data can be passed directly into VietCulturalDataset.
    """
    split_filename = _resolve_split_filename(split)

    print("Scanning Hugging Face dataset files...")
    repo_file_set = set(list_repo_files(repo_id=repo_id, repo_type="dataset"))

    if split_filename not in repo_file_set:
        raise FileNotFoundError(
            f"Could not find '{split_filename}' in dataset repo '{repo_id}'."
        )

    print(f"Downloading JSON file for split '{split}'...")
    json_path = hf_hub_download(
        repo_id=repo_id,
        filename=split_filename,
        repo_type="dataset",
    )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if n_samples is not None:
        data = data[:n_samples]

    base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
    print(f"Loaded {len(data)} samples. Processing images...")

    success_count = 0
    failed_count = 0

    for sample in tqdm(data, total=len(data), desc=f"Processing {split} images"):
        raw_image_path = sample.get("image_path", "")
        resolved_path = resolve_image_path(raw_image_path)

        if not resolved_path:
            sample["image"] = None
            failed_count += 1
            continue

        if resolved_path not in repo_file_set:
            sample["image"] = None
            failed_count += 1
            continue

        image_url = base_url + resolved_path

        try:
            sample["image"] = _download_image(image_url, timeout=request_timeout)
            success_count += 1
        except Exception:
            sample["image"] = None
            failed_count += 1

    print("\nProcessing images done.")
    print(f"Successful images: {success_count}")
    print(f"Failed images: {failed_count}")

    return [sample for sample in data if sample.get("image") is not None]


def build_viet_cultural_dataset(
    split="train",
    repo_id=DEFAULT_REPO_ID,
    n_samples=None,
    image_size=None,
    normalize=False,
    request_timeout=30,
):
    data = load_viet_cultural_split(
        split=split,
        repo_id=repo_id,
        n_samples=n_samples,
        request_timeout=request_timeout,
    )

    return VietCulturalDataset(
        data,
        image_size=image_size,
        normalize=normalize,
    )


def build_viet_cultural_datasets(
    repo_id=DEFAULT_REPO_ID,
    train_samples=None,
    val_samples=None,
    test_samples=None,
    image_size=None,
    normalize=False,
    request_timeout=30,
):
    return {
        "train": build_viet_cultural_dataset(
            split="train",
            repo_id=repo_id,
            n_samples=train_samples,
            image_size=image_size,
            normalize=normalize,
            request_timeout=request_timeout,
        ),
        "validation": build_viet_cultural_dataset(
            split="validation",
            repo_id=repo_id,
            n_samples=val_samples,
            image_size=image_size,
            normalize=normalize,
            request_timeout=request_timeout,
        ),
        "test": build_viet_cultural_dataset(
            split="test",
            repo_id=repo_id,
            n_samples=test_samples,
            image_size=image_size,
            normalize=normalize,
            request_timeout=request_timeout,
        ),
    }


class VietCulturalDataset(Dataset):
    def __init__(self, data_list, image_size=None, normalize=False):
        self.data = data_list
        self.samples = []
        self.normalize = normalize
        self.image_size = image_size

        for image_sample in data_list:
            image = image_sample.get("image")
            image_path = image_sample.get("image_path", "")

            question_list = image_sample.get("questions", [])

            if image is None:
                continue

            for qa in question_list:
                self.samples.append({
                    "image_path": image_path,
                    "image": image,
                    "question_id": qa.get("question_id"),
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "detailed_explanation": qa.get("detailed_explanation", ""),
                    "cultural_significance": qa.get("cultural_significance", ""),
                    "additional_context": qa.get("additional_context", {}),
                    "difficulty": qa.get("difficulty", ""),
                    "question_type": qa.get("question_type", ""),
                    "cognitive_level": qa.get("cognitive_level", ""),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample["image"]

        if self.image_size is not None:
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize(self.image_size)
            image = np.array(pil_image)

        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)

        if self.normalize:
            image_tensor = image_tensor.float() / 255.0

        return {
            "image_path": sample["image_path"],
            "image": image_tensor,
            "question_id": sample["question_id"],
            "question": sample["question"],
            "answer": sample["answer"],
            "detailed_explanation": sample["detailed_explanation"],
            "cultural_significance": sample["cultural_significance"],
            "additional_context": sample["additional_context"],
            "difficulty": sample["difficulty"],
            "question_type": sample["question_type"],
            "cognitive_level": sample["cognitive_level"],
        }
