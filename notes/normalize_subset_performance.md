# Notes on trying to get `normalize_subset` to work

# TODOs
- ENSURE DETERMINISTIC PERFORMANCE
- Need to make linkages possible after discounting punctuation, capitalization, etc. (maybe context-based stopwords too, such as "Co.", etc.).
    - ~~Low hanging fruit: just uppercase CUSIP, ISIN, FIGI before processing~~