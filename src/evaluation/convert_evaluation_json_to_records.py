import json
from pathlib import Path
from typing import Any


INPUT_PATH = Path("func_calling_dataset_gemma_2_2b_it_responses.json")
OUTPUT_PATH = Path("func_calling_dataset_gemma_2_2b_it_responses_records.json")


def main() -> None:
    with INPUT_PATH.open(encoding="utf-8") as input_file:
        columns: Any = json.load(input_file)

    if not isinstance(columns, dict) or not columns:
        raise ValueError(f"{INPUT_PATH} must contain a non-empty JSON object")

    if not all(isinstance(values, dict) for values in columns.values()):
        raise ValueError("Expected each top-level column to map row indices to values")

    first_column = next(iter(columns.values()))
    indices = list(first_column)
    expected_indices = set(indices)

    for name, values in columns.items():
        if set(values) != expected_indices:
            raise ValueError(f"Column {name!r} has different row indices")

    try:
        indices.sort(key=int)
    except ValueError as error:
        raise ValueError("Row indices must be numeric strings") from error

    records = [
        {name: values[index] for name, values in columns.items()}
        for index in indices
    ]

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(records, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")

    print(f"Wrote {len(records)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
