"""
config.py

This file contains global configurations, including:
- File paths for dataset storage.
- Constants for CSV field limits.

Usage:
    Import this module in other scripts to access file paths and settings.

Example:
    from config import OUTPUT_CSV, PROCESSED_CSV
"""

import sys
import csv

# File paths
DIR_TARGET = "/Users/odedlaufer/Documents/programming/python/information_retrieval/text_classification/3d_printing_txt"
DIR_RANDOM = "/Users/odedlaufer/Documents/programming/python/information_retrieval/text_classification/indoor_positioning_in_cultural_heritage_txt"
OUTPUT_CSV = "labeled_dataset.csv"
PROCESSED_CSV = "labeled_dataset_processed.csv"

# Increase CSV field size limit to prevent truncation issues
csv.field_size_limit(sys.maxsize)
