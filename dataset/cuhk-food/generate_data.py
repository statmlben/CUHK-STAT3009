"""Build a classroom CUHK canteen dataset with six food-preference archetypes.

Menu items, prices, and canteen descriptions come from the public
Lo-Hei-Yip/CUHK-CanteenDish-Rating-Recommendation repository.
Users and ratings are synthetic; each user is generated from one of six
latent food-preference directions so that a K=2 MF model has an interpretable
structure. No real student data is used.
"""
from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "menu_source.json"
SEED = 3009

CANTEEN_META = {
    "New Asia College Student Canteen": ("new-asia-canteen.jpg", None, None),
    "United College Student Canteen": ("united-canteen.jpg", None, None),
    "Chung Chi College Student Canteen": ("chung-chi-canteen.jpg", "23555", (3.37, 30)),
    "CUHK Central Canteen": ("cuhk-central-canteen.jpg", None, None),
    "CUHK Noodle House": ("cuhk-noodle-house.jpg", None, None),
    "Pommerenke Student Centre Coffee Shop": ("pommerenke-coffee.jpg", None, None),
    "CUHK Paper & Coffee": ("paper-coffee.jpg", "589722", (4.32, 31)),
    "Orchid Lodge Restaurant": ("orchid-lodge.jpg", "31254", (3.91, 23)),
}

CATEGORY_RULES = [
    ("Vegetarian", ("素", "野菜", "雜菜", "杂菜", "蔬菜", "豆腐")),
    ("Noodles", ("米線", "米线", "烏冬", "乌冬", "拉麵", "拉面", "河粉", "意粉", "通粉", "撈丁", "捞丁", "麵", "面")),
    ("Rice", ("飯", "饭", "丼", "定食", "煲", "焗", "粥")),
    ("Burgers & Sandwiches", ("漢堡", "汉堡", "熱狗", "热狗", "三文治", "牛角飽", "牛角饱", "包")),
    ("Salad", ("沙律", "沙拉")),
    ("Snacks", ("薯條", "薯条", "小食", "雞翼", "鸡翼", "魚蛋", "鱼蛋", "燒賣", "烧卖")),
    ("Cafe & Dessert", ("咖啡", "蛋糕", "西多士", "雪糕", "甜品", "多士", "鬆餅", "松饼", "麥皮", "麦皮")),
]
MEAT_TERMS = ("雞", "鸡", "豬", "猪", "牛", "魚", "鱼", "肉", "海鮮", "海鲜", "蝦", "虾",
              "扎肉", "煙三文魚", "烟三文鱼", "火腿", "腸", "肠", "午餐肉", "排骨", "扒")
SPICY_TERMS = ("辣", "川", "花椒", "泡菜", "韓式", "韩式")
FRIED_TERMS = ("炸", "吉列", "天婦羅", "天妇罗")
HEALTHY_TERMS = ("沙律", "沙拉", "素", "野菜", "雜菜", "杂菜", "湯", "汤", "蒸")

GROUPS = [
    {"key": "Noodles", "label": "Noodle explorer", "angle": 0.0,
     "categories": {"Noodles"}},
    {"key": "Rice", "label": "Rice bowl regular", "angle": math.pi / 3,
     "categories": {"Rice"}},
    {"key": "Burgers & Sandwiches", "label": "Burger & sandwich fan", "angle": 2 * math.pi / 3,
     "categories": {"Burgers & Sandwiches"}},
    {"key": "Cafe & Dessert", "label": "Cafe & dessert", "angle": math.pi,
     "categories": {"Cafe & Dessert"}},
    {"key": "Vegetarian & lighter", "label": "Vegetarian & lighter", "angle": 4 * math.pi / 3,
     "categories": {"Vegetarian", "Salad"}},
    {"key": "Snacks", "label": "Snack run", "angle": 5 * math.pi / 3,
     "categories": {"Snacks"}},
]


def has_any(name: str, terms: tuple[str, ...]) -> bool:
    return any(term in name for term in terms)


def classify(name: str) -> str:
    for category, terms in CATEGORY_RULES:
        if any(term in name for term in terms):
            if category == "Vegetarian" and has_any(name, MEAT_TERMS):
                continue
            return category
    return "Others"


def unit_vector(angle: float, noise: float) -> tuple[float, float]:
    return (math.cos(angle) + random.gauss(0, noise),
            math.sin(angle) + random.gauss(0, noise))


def main() -> None:
    random.seed(SEED)
    source = json.loads(SOURCE.read_text())
    raw_dishes = source["dishes"]

    canteens = []
    for canteen_id, (name, (image, openrice_id, openrice)) in enumerate(CANTEEN_META.items()):
        canteens.append({
            "canteen_id": canteen_id,
            "canteen": name,
            "short_name": name.replace(" College Student Canteen", "").replace(" Student Centre Coffee Shop", ""),
            "image": f"images/{image}",
            "openrice_id": openrice_id or "",
            "openrice_taste": f"{openrice[0]:.2f}" if openrice else "",
            "openrice_reviews": openrice[1] if openrice else "",
        })
    canteen_ids = {row["canteen"]: row["canteen_id"] for row in canteens}

    category_angle = {}
    for group in GROUPS:
        for category in group["categories"]:
            category_angle[category] = group["angle"]

    dishes = []
    for dish_id, item in enumerate(raw_dishes):
        name = item["name"]
        price = float(item["price"])
        category = classify(name)
        angle = category_angle.get(category, random.uniform(0, 2 * math.pi))
        vector = unit_vector(angle, 0.08)
        dishes.append({
            "dish_id": dish_id,
            "dish": name,
            "canteen_id": canteen_ids[item["canteen"]],
            "canteen": item["canteen"],
            "price": f"{price:.1f}",
            "category": category,
            "spicy": int(has_any(name, SPICY_TERMS)),
            "fried": int(has_any(name, FRIED_TERMS)),
            "healthy": int(has_any(name, HEALTHY_TERMS)),
            "_vector": vector,
            "_bias": random.gauss(0, 0.12),
        })

    group_sequence = []
    for index in range(6):
        group_sequence.extend([GROUPS[index]] * (54 if index < 2 else 53))
    random.shuffle(group_sequence)

    users = []
    for user_id, group in enumerate(group_sequence):
        vector = unit_vector(group["angle"], 0.10)
        users.append({
            "user_id": user_id,
            "food_preference": group["key"],
            "_activity": random.randint(22, 55),
            "_generosity": round(random.gauss(0, 0.12), 3),
            "_vector": vector,
            "_group_categories": group["categories"],
        })

    rows = []
    for user in users:
        group_categories = user["_group_categories"]
        weights = [6 if dish["category"] in group_categories else 1 for dish in dishes]
        picked = set()
        while len(picked) < user["_activity"]:
            if random.random() < 0.8:
                dish = random.choices(dishes, weights=weights, k=1)[0]
            else:
                dish = random.choice(dishes)
            if dish["dish_id"] in picked:
                continue
            picked.add(dish["dish_id"])
            latent = user["_vector"][0] * dish["_vector"][0] + user["_vector"][1] * dish["_vector"][1]
            score = 3.75 + 1.15 * latent + dish["_bias"] + user["_generosity"] + random.gauss(0, 0.28)
            score = max(1.0, min(5.0, score))
            rows.append({
                "user_id": user["user_id"],
                "dish_id": dish["dish_id"],
                "rating": f"{round(score * 2) / 2:.1f}",
            })

    # 80/20 split by user, then make sure every dish appears in train.
    by_user = defaultdict(list)
    by_dish = defaultdict(list)
    for row in rows:
        by_user[row["user_id"]].append(row)
        by_dish[row["dish_id"]].append(row)
    test_ids = set()
    for group in by_user.values():
        n_test = max(1, round(0.2 * len(group)))
        for row in random.sample(group, min(n_test, len(group) - 1)):
            test_ids.add(id(row))
    for dish_id, group in by_dish.items():
        if all(id(row) in test_ids for row in group):
            test_ids.remove(id(random.choice(group)))
    for row in rows:
        row["split"] = "test" if id(row) in test_ids else "train"

    fields = ["user_id", "dish_id", "rating", "split"]
    with (ROOT / "ratings.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with (ROOT / "users.csv").open("w", newline="") as f:
        fields = ["user_id", "food_preference"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: v for k, v in user.items() if not k.startswith("_")} for user in users])

    with (ROOT / "dishes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in dishes[0] if not k.startswith("_")])
        writer.writeheader()
        writer.writerows([{k: v for k, v in dish.items() if not k.startswith("_")} for dish in dishes])

    with (ROOT / "canteens.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(canteens[0]))
        writer.writeheader()
        writer.writerows(canteens)

    group_counts = Counter(user["food_preference"] for user in users)
    print(f"wrote {len(rows)} ratings, {len(users)} users, {len(dishes)} dishes, {len(canteens)} canteens")
    print("food preferences:", dict(group_counts))
    print("rating mean:", round(sum(float(r["rating"]) for r in rows) / len(rows), 3))


if __name__ == "__main__":
    main()
