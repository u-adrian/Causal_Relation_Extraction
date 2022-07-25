# Causal_Relation_Extraction

To train baseline model:
  1. Go into folder ```baseline model```
  2. Install requirements
  3. run ```python main.py --dataset_path "data/crest_1_2_13.xlsx"``` or ```python main.py --dataset_path "data/semeval_t8.xlsx" ```

To fine-tune BERT model:
  1. Go into folder ```BERT-Relation-Extraction```
  2. Install requirements
  3. run ```python main_task.py --task "semeval" --num_classes 19 --train 1 --infer 1``` or ```python main_task.py --task "crest" --num_classes 3 --train 1 --infer 1```

Bonus:
  1. See folder ```Stanford-NLP-Bonus```
  2. Install requirements
  3. Start notebook
