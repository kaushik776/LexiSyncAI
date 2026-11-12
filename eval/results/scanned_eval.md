The same 25 questions (21 answerable, 4 unanswerable) asked of each copy of the SEC exhibit, searching only that copy. Gold pages are identical, so the difference between the rows is the effect of OCR.

**Retrieval** (no LLM)

| Copy | Configuration | R@1 | R@3 | R@5 | R@10 | MRR | Multi-hop all@5 |
|---|---|---|---|---|---|---|---|
| Original (digital) | BM25 only | 0.810 | 0.905 | 0.929 | 1.000 | 0.934 | 0.800 |
| Original (digital) | Dense only | 0.690 | 0.833 | 0.905 | 0.952 | 0.841 | 0.800 |
| Original (digital) | Hybrid (BM25 + dense, RRF) | 0.738 | 0.905 | 0.952 | 1.000 | 0.889 | 1.000 |
| Original (digital) | Hybrid + refs + rerank (final) | 0.786 | 0.929 | 0.929 | 0.929 | 0.929 | 0.800 |
| Scanned (OCR) | BM25 only | 0.762 | 0.905 | 0.952 | 0.952 | 0.905 | 1.000 |
| Scanned (OCR) | Dense only | 0.524 | 0.738 | 0.762 | 0.810 | 0.659 | 0.800 |
| Scanned (OCR) | Hybrid (BM25 + dense, RRF) | 0.548 | 0.786 | 0.857 | 0.952 | 0.731 | 1.000 |
| Scanned (OCR) | Hybrid + refs + rerank (final) | 0.738 | 0.929 | 0.929 | 0.976 | 0.910 | 0.800 |

**Answers** (full system: hybrid retrieval, rerank, planner, verifier)

| Copy | Key facts | Key facts (multi-hop) | Citation precision | Evidence recall | Abstention acc. | False abstention | Mean s |
|---|---|---|---|---|---|---|---|
| Original (digital) | 0.77 | 0.767 | 0.858 | 0.833 | 1.0 | 0.143 | 10.235 |
| Scanned (OCR) | 0.857 | 1.0 | 0.86 | 0.881 | 1.0 | 0.095 | 10.466 |

Per question (key-fact recall and badge):

| Question | Original (digital) | Scanned (OCR) |
|---|---|---|
| F01 What was the original principal amount of the Secured Convertible Prom | 1.00 (HIGH) | 1.00 (HIGH) |
| F02 What is the new Mandatory Monthly Prepayment Amount under Amendment #2 | abstained (INSUFFICIENT EVIDENCE) | 1.00 (MEDIUM) |
| F03 To what amount is the Floor Price reduced by the amendment? | 1.00 (HIGH) | 0.00 (HIGH) |
| F04 By what percentage is the Outstanding Balance increased, and what is t | 1.00 (HIGH) | 1.00 (HIGH) |
| F05 What is the maximum percentage of the outstanding Common Shares that t | 1.00 (HIGH) | 1.00 (HIGH) |
| F06 How much subordinated debt may the Company take on, and from how many  | 1.00 (HIGH) | 1.00 (HIGH) |
| F07 How many Pre-Delivery Shares does the Investor buy, and at what price  | 1.00 (HIGH) | 1.00 (LOW) |
| F08 How does the amendment change the share reserve held for the Investor  | 1.00 (HIGH) | 1.00 (HIGH) |
| F09 Who signed Amendment #2 on behalf of the Investor and on behalf of the | 1.00 (LOW) | 1.00 (LOW) |
| F10 How large is the Organizational Fee paid to AMH under the Patent Monet | 1.00 (HIGH) | 1.00 (HIGH) |
| F11 What Management Fee does AMH receive, and for how long? | 1.00 (HIGH) | 1.00 (LOW) |
| F12 How are proceeds from licensing the Patents to a person other than a D | 0.33 (HIGH) | 1.00 (HIGH) |
| F13 How long is the Option Period in which AMH may decide to fund a Subseq | 1.00 (HIGH) | 1.00 (HIGH) |
| F14 Which state's law governs the Patent Monetization Agreement? | abstained (INSUFFICIENT EVIDENCE) | abstained (INSUFFICIENT EVIDENCE) |
| F15 Where are mediation and arbitration proceedings under the Patent Monet | 1.00 (HIGH) | abstained (INSUFFICIENT EVIDENCE) |
| F16 For how long can the option to buy the membership interests in Alpha M | abstained (INSUFFICIENT EVIDENCE) | 1.00 (HIGH) |
| M01 If the Company exercises its option to purchase the membership interes | 0.33 (HIGH) | 1.00 (HIGH) |
| M02 Which agreement with Alpha Modus Ventures does the Investor consent to | 1.00 (HIGH) | 1.00 (HIGH) |
| M03 How are disputes under the Patent Monetization Agreement resolved, and | 0.50 (LOW) | 1.00 (LOW) |
| M04 What deadlines after the Amendment Date does the Company have for the  | 1.00 (HIGH) | 1.00 (HIGH) |
| M05 Who are the members of Alpha Modus Ventures that grant the option, and | 1.00 (LOW) | 1.00 (LOW) |
| U01 What interest rate does the Secured Convertible Promissory Note carry? | abstained (INSUFFICIENT EVIDENCE) | abstained (INSUFFICIENT EVIDENCE) |
| U02 What is the maturity date of the Note? | abstained (INSUFFICIENT EVIDENCE) | abstained (INSUFFICIENT EVIDENCE) |
| U03 Which patent numbers are listed in Schedule A of the Patent Monetizati | abstained (INSUFFICIENT EVIDENCE) | abstained (INSUFFICIENT EVIDENCE) |
| U04 What is the total Litigation Budget for the initial Identified Litigat | abstained (INSUFFICIENT EVIDENCE) | abstained (INSUFFICIENT EVIDENCE) |
