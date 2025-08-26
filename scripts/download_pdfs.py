#!/usr/bin/env python3
"""
download_pdfs.py

Search and download full-text PDFs of papers from arXiv (or OpenAlex).
Save each PDF to a directory with filename '[DOI].pdf'.

Usage:
    python download_pdfs.py \
        --keyword "organic photovoltaics" \
        --target ./pdfs \
        --max-results 100
"""

import os
import time
import argparse
import logging
import arxiv
import requests
from pyalex import Works

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def search_arxiv(keyword: str, max_results: int):
    """
    Search arXiv using arxiv.py wrapper.
    Returns a list of arxiv.Result objects.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=keyword,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )
    return client.results(search)


def download_pdf(url: str, dest: str, retries=3):
    """
    Download PDF from URL and save to dest file.
    Handles retries, timeout, streaming.
    """
    for attempt in range(1, retries + 1):
        try:
            logging.debug(f"Downloading {url}")
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            with open(dest, 'wb') as f:
                for chunk in resp.iter_content(1024*32):
                    if chunk:
                        f.write(chunk)
            logging.info(f"Saved PDF to {dest}")
            return True
        except Exception as e:
            logging.warning(f"Attempt {attempt} failed: {e}")
            time.sleep(5)
    logging.error(f"Failed to download {url}")
    return False


def run_arxiv_workflow(keyword: str, target_dir: str, max_results: int):
    os.makedirs(target_dir, exist_ok=True)
    for result in search_arxiv(keyword, max_results):
        doi = (result.doi or result.get_short_id()).replace('/', '_')
        pdf_url = result.pdf_url
        filename = os.path.join(target_dir, f"{doi}.pdf")
        if os.path.exists(filename):
            logging.info(f"Already exists: {filename}")
            continue
        success = download_pdf(pdf_url, filename)
        time.sleep(3)  # respect rate limits


def run_openalex_workflow(keyword: str, target_dir: str, max_results: int):
    os.makedirs(target_dir, exist_ok=True)
    works = Works().search(title=keyword).per_page(max_results).execute()
    for item in works:
        doi = item.doi.replace('/', '_') if item.doi else None
        url = item.primary_location.pdf_url if item.primary_location else None
        if not doi or not url:
            logging.debug(f"Skipping missing DOI/pdf: {item.id}")
            continue
        filename = os.path.join(target_dir, f"{doi}.pdf")
        if os.path.exists(filename):
            continue
        download_pdf(url, filename)
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keyword', required=True)
    parser.add_argument('--target', default='./pdfs')
    parser.add_argument('--max-results', type=int, default=500)
    parser.add_argument('--source', choices=['arxiv', 'openalex'], default='arxiv')
    args = parser.parse_args()

    logging.info(f"Starting download: {args.keyword}")
    if args.source == 'openalex':
        run_openalex_workflow(args.keyword, args.target, args.max_results)
    else:
        run_arxiv_workflow(args.keyword, args.target, args.max_results)

if __name__ == '__main__':
    main()
