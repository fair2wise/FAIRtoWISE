# Find & Download PDFs

You can scrape arXiv and OpenAlex for PDFs relevant to your topic using this program:
 `scripts/download_pdfs.py`

It will search Arxiv based on your provided keyword/topic, sort by relevance, and download the number of PDFs you specific to a directory with filename '[DOI].pdf'. If you later decide you want to download more papers, you can increase the `--max-results` parameter and it will skip papers that have already been downloaded.

Usage:
```python
python download_pdfs.py \
    --keyword "organic photovoltaics" \
    --target ./pdfs \
    --max-results 100
```

# Annotate Python Scripts with NVTX

`nvtx_annotate.py`

Safely adds `@annotate` decorators and inserts `from nvtx import annotate` after the module docstring and existing imports,
preserving original formatting and docstrings. The original file is overwritten, but an `original_script.py.bak` is also retained for backup.

Usage:
```python
python nvtx_annotate.py <your_script.py>
```