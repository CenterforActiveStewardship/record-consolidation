<!-- # record-consolidation -->
<!-- 
## Overview

`record-consolidation` is designed to consolidate/unify data records across linkages.
Given a polars DataFrame of disagreeing records, `record-consolidation` exploits their natural linkage structures to normalize & consolidate the data.

Under the hood, `record-consolidation` converts input dataframes into a `networkx` graph, extracts the strongly connected components as groups, determines a "canonical" identity for each group, and and assigns this canonical identity back into the original dataframe

`record-consolidation` provides options for augmentating this process. Most notably, users can provide their own function for processing graphs before "canonical" identities are assigned to each group. For example, a specific partitioning algorithm (such as the included `partition_subgraphs()`) can be passed - this is especially useful if entities are sometimes spuriously recorded with the identifiers from another group, leading to weakly linked clusters that should be partitioned.

## Examples

<!-- ![alt text](image.png) -->
<!-- <img src="image.png" alt="drawing" width="200"/> -->

<!-- <div style="text-align: center;">
    <img src="image.png" alt="drawing" width="200"/>
    <a href="https://www.vecteezy.com/free-vector/disc" style="font-size: 8px;">
  Disc Vectors by Vecteezy
</a> -->

<div style="text-align: center;">
    <img src="image.png" alt="drawing" width="200" style="display: block; margin: 0 auto;" />
</div>

</div>

# record-consolidation: Record Unification in Python

`record-consolidation` is a Python package for consolidating data records with erratic linkages, powered by a graph-based backend and customizable processing options.

- Natural linkage-based consolidation
- Flexible graph partitioning
- Built with `polars` and `networkx` for speed and scalability

## Overview

`record-consolidation` allows users to unify datasets with inconsistent or fragmented records by leveraging the frequency of linkages between fields. Given a DataFrame of records with discrepancies, `record-consolidation` maps these records to a `networkx` graph, groups strongly connected records, and uses the resulting strongly connected components to determine a "canonical" identity for each group. `record-consolidation` provides functionality for assigning these identities back to the original DataFrame, overwriting incorrect/missing identifiers.

Options for customization are available, allowing users to inject specific processing functions or partitioning methods to refine the consolidation process. For example, weakly linked clusters can be split using custom algorithms when identifiers are spuriously shared between unrelated records (an example/default is included: `partition_subgraphs`).

## Installation
```sh
# rye # TODO: test
rye add record-consolidation --git https://github.com/m-gaster/record-consolidation/

# uv # TODO: test
uv add record-consolidation git+https://github.com/m-gaster/record-consolidation/

# pip # TODO: test
pip install git+https://github.com/m-gaster/record-consolidation.git

# poetry # TODO: test
poetry add record-consolidation --git https://github.com/m-gaster/record-consolidation.git
```

## Examples
### Example: Normalize across all fields
Take this example of disagreeing records: "Tom" should have `id == 12` for all observations, but observations number 7 and 8 have values `null` and `0`, respectively.

```python
>>> import polars as pl
>>> from record_consolidation import normalize_subset

>>> tom_records = pl.DataFrame({
...     "name": ["Tom"] * 10,
...     "id": [12] * 7 + [None, 0, 12],
... })
>>> tom_records
shape: (10, 2)
┌──────┬──────┐
│ name ┆ id   │
│ ---  ┆ ---  │
│ str  ┆ i64  │
╞══════╪══════╡
│ Tom  ┆ 12   │
│ Tom  ┆ 12   │
│ Tom  ┆ 12   │
│ Tom  ┆ 12   │
│ Tom  ┆ 12   │
│ Tom  ┆ 12   │
│ Tom  ┆ 12   │
│ Tom  ┆ null │
│ Tom  ┆ 0    │
│ Tom  ┆ 12   │
└──────┴──────┘


# Normalize the subset to correct the inconsistencies
>>> normalize_subset(tom_records, cols_to_normalize="all")
shape: (10, 2)
┌──────┬─────┐
│ name ┆ id  │
│ ---  ┆ --- │
│ str  ┆ i64 │
╞══════╪═════╡
│ Tom  ┆ 12  │
│ Tom  ┆ 12  │
│ Tom  ┆ 12  │
│ Tom  ┆ 12  │
│ Tom  ┆ 12  │
│ Tom  ┆ 12  │
│ Tom  ┆ 12  │
│ Tom  ┆ 12  │
│ Tom  ┆ 12  │
│ Tom  ┆ 12  │
└──────┴─────┘
```

### Example: Normalize only a subset of fields
Some fields may vary independently of identifiers and shouldn’t be normalized. In this example, we limit normalization to `"name"` and `"id"` only.
```python
>>> tom_records = pl.DataFrame({
...     "name": ["Tom"] * 10,
...     "id": [12] * 7 + [None, 0, 12],
...     "age": [50] * 5 + [51] * 5,
...     "year_recorded": [2023] * 5 + [2024] * 5,
... })
>>> tom_records
shape: (10, 4)
┌──────┬──────┬─────┬───────────────┐
│ name ┆ id   ┆ age ┆ year_recorded │
│ ---  ┆ ---  ┆ --- ┆ ---           │
│ str  ┆ i64  ┆ i64 ┆ i64           │
╞══════╪══════╪═════╪═══════════════╡
│ Tom  ┆ 12   ┆ 50  ┆ 2023          │
│ Tom  ┆ 12   ┆ 50  ┆ 2023          │
│ Tom  ┆ 12   ┆ 50  ┆ 2023          │
│ Tom  ┆ 12   ┆ 50  ┆ 2023          │
│ Tom  ┆ 12   ┆ 50  ┆ 2023          │
│ Tom  ┆ 12   ┆ 51  ┆ 2024          │
│ Tom  ┆ 12   ┆ 51  ┆ 2024          │
│ Tom  ┆ null ┆ 51  ┆ 2024          │
│ Tom  ┆ 0    ┆ 51  ┆ 2024          │
│ Tom  ┆ 12   ┆ 51  ┆ 2024          │
└──────┴──────┴─────┴───────────────┘

# normalize only ["name", "id"]
>>> normalize_subset(
...     tom_records,
...     cols_to_normalize=["name", "id"],
)
shape: (10, 4)
┌──────┬─────┬─────┬───────────────┐
│ name ┆ id  ┆ age ┆ year_recorded │
│ ---  ┆ --- ┆ --- ┆ ---           │
│ str  ┆ i64 ┆ i64 ┆ i64           │
╞══════╪═════╪═════╪═══════════════╡
│ Tom  ┆ 12  ┆ 50  ┆ 2023          │
│ Tom  ┆ 12  ┆ 50  ┆ 2023          │
│ Tom  ┆ 12  ┆ 50  ┆ 2023          │
│ Tom  ┆ 12  ┆ 50  ┆ 2023          │
│ Tom  ┆ 12  ┆ 50  ┆ 2023          │
│ Tom  ┆ 12  ┆ 51  ┆ 2024          │
│ Tom  ┆ 12  ┆ 51  ┆ 2024          │
│ Tom  ┆ 12  ┆ 51  ┆ 2024          │
│ Tom  ┆ 12  ┆ 51  ┆ 2024          │
│ Tom  ┆ 12  ┆ 51  ┆ 2024          │
└──────┴─────┴─────┴───────────────┘
```

### Example: Extract atomized records
`atomize_subset` will extract all canonical identities from records.
```python
from record_consolidation import atomize_subset

>>> atomize_subset(
...     tom_record.select(["name", "id"]),
...     pre_processing_fnc=None,
...     connected_subgraphs_postprocessor=None,
... )
┌──────┬─────┐
│ name ┆ id  │
│ ---  ┆ --- │
│ str  ┆ i64 │
╞══════╪═════╡
│ Tom  ┆ 12  │
└──────┴─────┘
```



<a href="https://www.vecteezy.com/free-vector/disc" style="font-size: 10px">
        Logo Attribution
</a>