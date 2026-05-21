import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def allocate_counts(total: int, train_ratio: float, val_ratio: float):
    raw = {
        "train": total * train_ratio,
        "val": total * val_ratio,
    }
    base = {split: math.floor(value) for split, value in raw.items()}
    base["test"] = total - base["train"] - base["val"]

    remainder = total - sum(base.values())
    if remainder > 0:
        ranked = sorted(
            raw.items(),
            key=lambda item: (item[1] - math.floor(item[1]), item[0]),
            reverse=True,
        )
        for split, _ in ranked[:remainder]:
            base[split] += 1
            base["test"] -= 1
    return base


def group_pairs_by_hr(pairs):
    groups = defaultdict(list)
    hr_to_source = {}
    hr_to_category = {}

    for pair in pairs:
        hr_image = pair["hr_image"]
        original_source = pair["original_source"]
        category = pair["category"]
        groups[hr_image].append(pair)

        if hr_image in hr_to_source and hr_to_source[hr_image] != original_source:
            raise ValueError(f"Inconsistent original_source for {hr_image}")
        if hr_image in hr_to_category and hr_to_category[hr_image] != category:
            raise ValueError(f"Inconsistent category for {hr_image}")

        hr_to_source[hr_image] = original_source
        hr_to_category[hr_image] = category

    return groups, hr_to_source, hr_to_category


def build_stratified_splits(groups, hr_to_category, train_ratio, val_ratio, seed):
    rng = random.Random(seed)
    category_to_hr = defaultdict(list)
    for hr_image, category in hr_to_category.items():
        category_to_hr[category].append(hr_image)

    split_hr = {"train": [], "val": [], "test": []}
    for category, hr_images in sorted(category_to_hr.items()):
        rng.shuffle(hr_images)
        counts = allocate_counts(len(hr_images), train_ratio, val_ratio)

        train_cut = counts["train"]
        val_cut = train_cut + counts["val"]

        split_hr["train"].extend(hr_images[:train_cut])
        split_hr["val"].extend(hr_images[train_cut:val_cut])
        split_hr["test"].extend(hr_images[val_cut:])

    split_pairs = {}
    for split_name, hr_images in split_hr.items():
        pairs = []
        for hr_image in hr_images:
            pairs.extend(groups[hr_image])
        split_pairs[split_name] = pairs
    return split_hr, split_pairs


def summarize(split_hr, split_pairs):
    summary = {
        "unique_hr_per_split": {k: len(v) for k, v in split_hr.items()},
        "pair_count_per_split": {k: len(v) for k, v in split_pairs.items()},
        "category_hr_per_split": {},
        "category_pairs_per_split": {},
        "difficulty_per_split": {},
    }

    for split_name, pairs in split_pairs.items():
        hr_counter = Counter()
        pair_counter = Counter()
        difficulty_counter = Counter()

        seen_hr = set()
        for pair in pairs:
            pair_counter[pair["category"]] += 1
            difficulty_counter[pair["degradation_metadata"]["difficulty_level"]] += 1
            if pair["hr_image"] not in seen_hr:
                hr_counter[pair["category"]] += 1
                seen_hr.add(pair["hr_image"])

        summary["category_hr_per_split"][split_name] = dict(hr_counter)
        summary["category_pairs_per_split"][split_name] = dict(pair_counter)
        summary["difficulty_per_split"][split_name] = dict(difficulty_counter)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Create corrected train/val/test splits grouped by HR image."
    )
    parser.add_argument(
        "--metadata-dir",
        default="metadata-final-dataset",
        help="Directory containing complete_metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="metadata-final-dataset-corrected",
        help="Directory to write corrected split JSON files.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Ratios must be positive and train_ratio + val_ratio must be < 1.")

    metadata_dir = Path(args.metadata_dir)
    output_dir = Path(args.output_dir)
    complete_metadata_path = metadata_dir / "complete_metadata.json"

    if not complete_metadata_path.exists():
        raise FileNotFoundError(f"Missing {complete_metadata_path}")

    complete_metadata = load_json(complete_metadata_path)
    pairs = complete_metadata["pairs"]

    groups, _, hr_to_category = group_pairs_by_hr(pairs)
    split_hr, split_pairs = build_stratified_splits(
        groups,
        hr_to_category,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    corrected_complete_metadata = dict(complete_metadata)
    corrected_complete_metadata["pairs"] = pairs
    corrected_complete_metadata["dataset_info"] = dict(corrected_complete_metadata.get("dataset_info", {}))
    corrected_complete_metadata["dataset_info"]["split_protocol"] = (
        "Grouped by HR image with category-stratified 70/15/15 split."
    )
    corrected_complete_metadata["dataset_info"]["split_seed"] = args.seed

    save_json(output_dir / "complete_metadata.json", corrected_complete_metadata)
    save_json(output_dir / "train_split.json", split_pairs["train"])
    save_json(output_dir / "val_split.json", split_pairs["val"])
    save_json(output_dir / "test_split.json", split_pairs["test"])
    save_json(output_dir / "split_summary.json", summarize(split_hr, split_pairs))

    report = {
        "output_dir": str(output_dir.resolve()),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "summary": summarize(split_hr, split_pairs),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
