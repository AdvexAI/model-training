## Script for Downstream model training

To evaluate model performance using Advex powered data:
1. Clone this repository and cd into `model-training`
2. Install packages: `python -r requirements.txt`
3. Run `python standalone_yolo.py --real-dir /path/to/customer_data --syn-dir /path/to/advex_data --epochs 200 --seed 1`




This script trains a model on customer data, another on customer data and affine augmentations, and third model on customer data and advex data. It will generate a report comparing
metrics such as F1, Recall, and Average Precision across the two models.

The report will look something like this:
```
--------------------------------
Evaluation results:
Customer Data:
  Average Precision: 0.0014
  Average Recall: 0.3333
  F1: 0.0033
Customer Data + Augmentations:
  Average Precision: 0.1021
  Average Recall: 0.3333
  F1: 0.2188
Customer Data + Advex:
  Average Precision: 0.4474
  Average Recall: 0.3333
  F1: 0.4888
```
