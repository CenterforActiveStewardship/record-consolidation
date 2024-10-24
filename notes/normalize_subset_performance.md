# Notes on trying to get `normalize_subset` to work

# TODOs
- Mattel is being erronesouly subsumed by Chevron.
    - This seems to be a clustering issue, rather than a merging one.
    - It's likely because there's a direct bridge between the Mattel and Chevron clusters, rather than an indirect double-bridge. 
- ENSURE DETERMINISTIC PERFORMANCE
- Need to make linkages possible after discounting punctuation, capitalization, etc. (maybe context-based stopwords too, such as "Co.", etc.).
    - ~~Low hanging fruit: just uppercase CUSIP, ISIN, FIGI before processing~~