| method                   |   n_retrieval |   page_recall@1 |   page_recall@3 |   page_recall@5 | r@5_95ci     |     mrr |   n_answers |   answer_accuracy | acc_95ci     |
|:-------------------------|--------------:|----------------:|----------------:|----------------:|:-------------|--------:|------------:|------------------:|:-------------|
| plain_dense              |           143 |           0.748 |           0.888 |           0.916 | 0.87 to 0.96 |   0.826 |      60.000 |             0.867 | 0.77 to 0.95 |
| plain_hybrid             |           143 |           0.650 |           0.860 |           0.937 | 0.90 to 0.97 |   0.770 |     nan     |           nan     | nan          |
| table_dense              |           143 |           0.720 |           0.860 |           0.895 | 0.85 to 0.94 |   0.800 |     nan     |           nan     | nan          |
| table_hybrid             |           143 |           0.678 |           0.839 |           0.916 | 0.87 to 0.96 |   0.781 |      60.000 |             0.850 | 0.75 to 0.93 |
| plain_dense+second_look  |             0 |         nan     |         nan     |         nan     | nan          | nan     |      60.000 |             0.900 | 0.82 to 0.97 |
| table_hybrid+second_look |             0 |         nan     |         nan     |         nan     | nan          | nan     |      60.000 |             0.900 | 0.82 to 0.97 |