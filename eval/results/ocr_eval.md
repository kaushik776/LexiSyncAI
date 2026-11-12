OCR on a real scanned document: `EX-10.3_scanned.pdf` (20 image-only pages), checked against its digital original `EX-10.3_original.pdf`.

| Metric | Value |
|---|---|
| Pages detected as scanned | 20/20 |
| OCR engine | easyocr |
| Character accuracy (mean per page) | 86.3% |
| Word accuracy (mean per page) | 78.0% |
| Worst page, word accuracy | 41.2% |
| OCR engine confidence (mean) | 77.5% |
| OCR time per page | 8.76 s |

Per page:

| Page | Scanned? | Engine | Char accuracy | Word accuracy | Confidence | Words (original / OCR) |
|---|---|---|---|---|---|---|
| 1 | yes | easyocr | 86.2% | 77.7% | 78.0% | 359 / 354 |
| 2 | yes | easyocr | 89.3% | 80.9% | 78.6% | 645 / 643 |
| 3 | yes | easyocr | 90.6% | 80.4% | 77.0% | 603 / 598 |
| 4 | yes | easyocr | 55.4% | 41.2% | 75.5% | 51 / 33 |
| 5 | yes | easyocr | 84.5% | 76.5% | 81.5% | 422 / 411 |
| 6 | yes | easyocr | 84.9% | 79.8% | 78.8% | 634 / 631 |
| 7 | yes | easyocr | 82.6% | 72.6% | 79.4% | 635 / 627 |
| 8 | yes | easyocr | 88.9% | 81.6% | 77.9% | 675 / 676 |
| 9 | yes | easyocr | 83.9% | 75.0% | 79.8% | 707 / 709 |
| 10 | yes | easyocr | 89.5% | 83.4% | 78.7% | 661 / 658 |
| 11 | yes | easyocr | 94.0% | 88.3% | 78.9% | 699 / 694 |
| 12 | yes | easyocr | 91.8% | 83.1% | 75.4% | 680 / 679 |
| 13 | yes | easyocr | 93.2% | 85.6% | 73.8% | 599 / 599 |
| 14 | yes | easyocr | 86.2% | 78.8% | 73.3% | 676 / 672 |
| 15 | yes | easyocr | 90.1% | 82.4% | 78.8% | 745 / 742 |
| 16 | yes | easyocr | 84.5% | 76.8% | 75.6% | 980 / 983 |
| 17 | yes | easyocr | 84.2% | 77.5% | 76.2% | 599 / 602 |
| 18 | yes | easyocr | 90.2% | 80.8% | 78.6% | 650 / 642 |
| 19 | yes | easyocr | 91.2% | 82.4% | 75.7% | 703 / 696 |
| 20 | yes | easyocr | 85.3% | 74.7% | 77.5% | 664 / 660 |

Page 4 is the signature page: few words in a two-column layout, so the OCR reads them in a different order and every short name that differs counts as an error.

Accuracy compares the text after the same cleaning on both versions (headers and footers removed, spaces and quote marks normalised), so it measures what retrieval and the answers see.
