import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_split_file(metadata_dir: Path, *names: str) -> Path | None:
    for name in names:
        path = metadata_dir / name
        if path.exists():
            return path
    return None


def summarize_complete_metadata(pairs):
    category_pairs = Counter()
    category_hr = defaultdict(set)
    difficulty_counts = Counter()
    variant_difficulty = defaultdict(Counter)
    hr_variant_counts = Counter()

    for pair in pairs:
        category = pair["category"]
        hr_image = pair["hr_image"]
        difficulty = pair["degradation_metadata"]["difficulty_level"]
        variant_index = pair["variant_index"]

        category_pairs[category] += 1
        category_hr[category].add(hr_image)
        difficulty_counts[difficulty] += 1
        variant_difficulty[variant_index][difficulty] += 1
        hr_variant_counts[hr_image] += 1

    return {
        "total_pairs": len(pairs),
        "unique_hr_images": len({pair["hr_image"] for pair in pairs}),
        "unique_original_sources": len({pair["original_source"] for pair in pairs}),
        "category_pair_counts": dict(category_pairs),
        "category_hr_counts": {k: len(v) for k, v in category_hr.items()},
        "difficulty_counts": dict(difficulty_counts),
        "variant_index_to_difficulty": {
            idx: dict(counts) for idx, counts in sorted(variant_difficulty.items())
        },
        "variants_per_hr_distribution": dict(Counter(hr_variant_counts.values())),
    }


def summarize_splits(split_data):
    summary = {}
    hr_to_splits = defaultdict(set)
    src_to_splits = defaultdict(set)
    hr_to_split_counts = defaultdict(Counter)

    for split_name, pairs in split_data.items():
        summary[split_name] = {
            "pair_count": len(pairs),
            "unique_hr_images": len({pair["hr_image"] for pair in pairs}),
            "unique_original_sources": len({pair["original_source"] for pair in pairs}),
            "category_pair_counts": dict(Counter(pair["category"] for pair in pairs)),
            "difficulty_counts": dict(
                Counter(pair["degradation_metadata"]["difficulty_level"] for pair in pairs)
            ),
        }

        for pair in pairs:
            hr_image = pair["hr_image"]
            original_source = pair["original_source"]
            hr_to_splits[hr_image].add(split_name)
            src_to_splits[original_source].add(split_name)
            hr_to_split_counts[hr_image][split_name] += 1

    overlap = {"hr_image": {}, "original_source": {}}
    split_names = list(split_data.keys())
    hr_sets = {
        split_name: {pair["hr_image"] for pair in pairs}
        for split_name, pairs in split_data.items()
    }
    src_sets = {
        split_name: {pair["original_source"] for pair in pairs}
        for split_name, pairs in split_data.items()
    }
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlap["hr_image"][f"{left}__{right}"] = len(hr_sets[left] & hr_sets[right])
            overlap["original_source"][f"{left}__{right}"] = len(
                src_sets[left] & src_sets[right]
            )

    hr_split_patterns = Counter(tuple(sorted(splits)) for splits in hr_to_splits.values())
    hr_variant_patterns = Counter(
        tuple((split_name, counts[split_name]) for split_name in sorted(counts))
        for counts in hr_to_split_counts.values()
    )

    summary["split_integrity"] = {
        "hr_images_spanning_multiple_splits": sum(
            1 for splits in hr_to_splits.values() if len(splits) > 1
        ),
        "original_sources_spanning_multiple_splits": sum(
            1 for splits in src_to_splits.values() if len(splits) > 1
        ),
        "hr_split_patterns": {
            ",".join(pattern): count for pattern, count in hr_split_patterns.items()
        },
        "hr_variant_distribution_across_splits": {
            str(pattern): count for pattern, count in hr_variant_patterns.items()
        },
        "pairwise_overlap": overlap,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Audit the published final metadata and split integrity."
    )
    parser.add_argument(
        "--metadata-dir",
        default="metadata-final-dataset",
        help="Directory containing complete_metadata.json and split JSON files.",
    )
    args = parser.parse_args()

    metadata_dir = Path(args.metadata_dir)
    complete_metadata_path = metadata_dir / "complete_metadata.json"
    train_split_path = resolve_split_file(metadata_dir, "train_split.json", "train-split.json")
    val_split_path = resolve_split_file(metadata_dir, "val_split.json", "val-split.json")
    test_split_path = resolve_split_file(metadata_dir, "test_split.json", "test-split.json")

    if not complete_metadata_path.exists():
        raise FileNotFoundError(f"Missing {complete_metadata_path}")
    if not all([train_split_path, val_split_path, test_split_path]):
        raise FileNotFoundError("Could not resolve one or more split files.")

    complete_metadata = load_json(complete_metadata_path)
    pairs = complete_metadata["pairs"]

    split_data = {
        "train": load_json(train_split_path),
        "val": load_json(val_split_path),
        "test": load_json(test_split_path),
    }

    report = {
        "metadata_dir": str(metadata_dir.resolve()),
        "complete_metadata_summary": summarize_complete_metadata(pairs),
        "split_summary": summarize_splits(split_data),
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
