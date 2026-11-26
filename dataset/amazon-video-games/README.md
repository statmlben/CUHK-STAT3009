# Amazon Reviews 2023 · Video Games subset

A small classroom subset of the **Amazon Reviews 2023** dataset (McAuley Lab,
UCSD). It is designed for the side-information notebook, with a continuous 1–5
star rating and four kinds of product signal:

- **continuous:** `price`, `user_count` (computed from training rows)
- **categorical:** `store`, `main_category`, `categories`
- **text:** `title`, `description`, `features` (encoded in the notebook)
- **image:** one product photo for 300 products (encoded in the notebook)

## Files

- `ratings.csv` — 39,796 ratings from 29,209 users on 2,000 products.
  Columns: `user_id`, `item_id`, `rating`, `timestamp`.
- `items.csv` — product metadata for the 2,000 products. Columns:
  `item_id`, `title`, `description`, `features`, `categories`,
  `main_category`, `store`, `price`, `image_path`.
- `images/` — 300 product images, resized to 224×224 JPEG. `image_path` in
  `items.csv` is relative to this folder.

The text and image embeddings are **not stored here**. `STAT3009_Side.ipynb`
computes them with Hugging Face pipelines:

- text: `sentence-transformers/all-MiniLM-L6-v2`
- image: `microsoft/resnet-18`

## How the subset was built

The source is the 5-core `rating_only` split for the Video Games category.
We selected 2,000 products with a title, non-empty description/features, a
finite price, and at least one image URL, then sampled up to 25 reviews per
product. The images were downloaded from the metadata URLs.

## Leakage warnings

- `average_rating` and `rating_number` are **rating-derived** and are excluded.
  They would leak the target and are unavailable for a genuinely new product.
- Review `title`/`text` are **not** used for rating prediction: the review is
  written together with the rating.
- For a new-item split, item IDs are unseen; only metadata, text, and images
  can represent the product.

## Source and citation

- Dataset: <https://amazon-reviews-2023.github.io/>
- Hugging Face: <https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023>
- Citation: Hou et al. (2024), *Bridging Language and Items for Retrieval and
  Recommendation*.

The subset is for classroom use. Check the original dataset license before
redistributing the images or metadata.
