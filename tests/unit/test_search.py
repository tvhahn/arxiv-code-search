"""Test search functions."""

import pandas as pd
from pathlib import Path
import numpy as np
import re
import os
import argparse
import logging
from datetime import datetime
import nltk
import shutil
from src.data.search_txt import create_chunks_of_text
import random
from random import randrange


import unittest


class TestSearch(unittest.TestCase):
    """Test search functions."""

    def test_create_chunks_of_text(self):

        random.seed(9)
        text_dummy = ". ".join(
            [f"{str(s)} " * randrange(0, 12) for s in np.arange(0, 45)]
        )

        split_dict_true = {
            0: [
                "0 0 0 0 0 0 0 . 1 1 1 1 1 1 1 1 1 . 2 2 2 2 2 . 3 3 3 3 . 4 4 . 5 5 .",
                35,
            ],
            1: [
                "5 5 . 6 6 6 6 6 6 6 6 6 6 . . 8 8 8 8 8 . 9 9 9 9 9 9 9 9 . 10 10 10 10 10 10 10 . 11 11 11 11 11 11 11 11 11 . 12 .",
                50,
            ],
            2: [
                "12 . 13 13 13 13 13 . 14 14 14 14 14 14 14 14 . 15 15 15 15 15 15 15 15 15 . 16 16 16 16 16 16 16 16 16 16 16 . . 18 18 18 18 18 18 18 18 18 18 18 .",
                52,
            ],
            3: [
                "19 19 19 19 19 19 . 20 20 . 21 21 21 21 21 21 21 21 21 21 21 . 22 22 22 22 22 22 22 . 23 23 23 23 23 23 23 23 23 23 23 . 24 24 24 24 24 24 .",
                49,
            ],
            4: [
                "23 23 23 23 23 23 23 23 23 23 23 . 24 24 24 24 24 24 . 25 25 . 26 26 . 27 27 27 . . 29 . 30 30 . 31 31 31 31 31 31 31 31 .",
                44,
            ],
            5: [
                "31 31 31 31 31 31 31 31 . 32 32 32 32 32 32 32 32 32 . 33 . 34 34 34 34 34 34 34 34 34 34 34 . 35 35 35 35 35 35 .",
                40,
            ],
            6: [
                "35 35 35 35 35 35 . 36 36 36 36 36 36 36 36 36 36 36 . 37 . 38 38 38 38 . 39 39 39 . 40 40 40 40 40 40 40 40 40 40 . 41 41 41 .",
                45,
            ],
            7: [
                "37 . 38 38 38 38 . 39 39 39 . 40 40 40 40 40 40 40 40 40 40 . 41 41 41 . 42 42 42 42 42 42 42 42 42 42 42 . 43 43 43 43 43 43 . 44",
                46,
            ],
        }

        split_dict = create_chunks_of_text(text_dummy, init_token_len=None, max_token_len=50)

        self.assertEqual(split_dict_true, split_dict)


if __name__ == "__main__":

    unittest.main()
