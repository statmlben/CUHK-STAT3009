# CUHK Canteen Explorer dataset

A small classroom dataset for EDA, matrix factorization, and a static recommendation demo.

## Files

- `dishes.csv` — 183 dishes across 8 CUHK canteens, with price, inferred category, and simple spicy/fried/healthy flags.
- `canteens.csv` — canteen names, short names, local photos, and OpenRice aggregate taste scores where a public match was available.
- `users.csv` — 320 **synthetic** student profiles, each assigned to one of six simulated food-preference archetypes (`food_preference` column).
- `ratings.csv` — 12,683 **synthetic** ratings with an 80/20 `train`/`test` split.
- `recommendations.json` — top-N dishes for six example profiles, plus campus favourites.
- `index.html` — self-contained static demo. Open it in a browser; no server is required.
- `menu_source.json` — the public menu snapshot used to build `dishes.csv`.
- `generate_data.py` — regenerates the synthetic users and ratings from the menu snapshot.
- `images/` — canteen photos from the public CUHK Canteen project.

## What is real?

The canteen names, dish names, prices, and canteen photos come from the public
[Lo-Hei-Yip/CUHK-CanteenDish-Rating-Recommendation](https://github.com/Lo-Hei-Yip/CUHK-CanteenDish-Rating-Recommendation)
repository. Three canteens also have aggregate OpenRice taste scores
(Chung Chi, Orchid Lodge, and Paper & Coffee) taken from the public
[OpenRice dataset](https://github.com/YanniYang-PolyU/OpenRice_dataset).

## What is simulated?

The 320 students and all 12,683 ratings are **synthetic**. Each user is generated
from one of six simulated food-preference archetypes (noodles, rice,
burgers/sandwiches, cafe/dessert, vegetarian/lighter, or snacks) plus
individual noise. They are not
real CUHK student records and must not be presented as evidence about real
campus preferences.

The synthetic ratings exist to make the classroom workflow reproducible:
EDA → train/test split → MF → cross-validation → top-N recommendations.

## Reproduce the data

From this directory:

```bash
python generate_data.py
```

The script is deterministic (`SEED = 3009`). Edit the profile distributions in
the script if you want a different classroom dataset.

## Open the demo

Open `index.html` directly in a browser, or serve this directory:

```bash
python -m http.server 8000
```

Then visit <http://localhost:8000/>.

The demo uses the real canteen photos and dish names, with synthetic predicted
ratings. It does not call any external API.
