# Products Recommender

This repository contains a Python implementation of a products recommendation engine using item-based collaborative filtering. The engine leverages PyNNDescent for efficient nearest neighbor search and Sentence Transformers for generation of embeddings and numerical representations.

## Key Features

- Item-based collaborative filtering
- PyNNDescent for efficient graph-based approximate nearest neighbor search
- Sentence Transformers for embedding generation

## Run script

```bash
python3.11 app.py <user_id>
```

NB: Based on the current dataset, user ids range from 1 - 1224.

## Dataset Acknowledgements

- This project utilises the Ukrainian Market Mobile Phones dataset, publicly available on Kaggle at [link](https://www.kaggle.com/datasets/artempozdniakov/ukrainian-market-mobile-phones-data/data). Sincere gratitude to the dataset creator(s) for making this valuable resource accessible to the public.

- Synthetic datasets are also generated to augment the public dataset.
