Table extraction vs manual ground truth (eval/tables_ground_truth.json).

| Table | Detected | Rows (pred/gold) | Shape match | Cell accuracy | Cell similarity | Numeric cells |
|---|---|---|---|---|---|---|
| EX-10.2.pdf p.25 | yes | 5/5 | yes | 1.000 | 1.000 | 1.0 |
| EX-10.2.pdf p.26 | yes | 5/5 | yes | 1.000 | 1.000 | 1.0 |
| UK_Consumer_Rights_Act_2015.pdf p.88 | yes | 7/7 | yes | 1.000 | 1.000 | 1.0 |
| UK_Consumer_Rights_Act_2015.pdf p.89 | yes | 2/2 | yes | 0.750 | 0.998 | 0.0 |
| **All** | 4/4 | | 4/4 | **0.981** | 1.000 | 0.75 |
