import math
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

# 1. Генерація датасету E-Commerce покупок із часовими мітками та профілями інтересів
NUM_USERS = 120
NUM_ITEMS = 80
CATEGORIES = ["Електроніка", "Фінтех-Кешбек", "Побутова техніка", "Спорт", "Книги", "Аксесуари"]
PRICE_TIERS = ["budget", "mid", "premium"]

# Атрибути товарів (для Content-based моделі)
items = {}
for i in range(NUM_ITEMS):
    cat = CATEGORIES[i % len(CATEGORIES)]
    tier = PRICE_TIERS[(i // len(CATEGORIES)) % len(PRICE_TIERS)]
    # Популярність розподілена за степеневим законом (Zipf / Long-tail)
    pop_weight = 1.0 / ((i + 1) ** 0.65)
    items[f"item_{i:02d}"] = {"cat": cat, "tier": tier, "pop_w": pop_weight}

item_ids = list(items.keys())
item_weights = np.array([items[iid]["pop_w"] for iid in item_ids])
item_weights /= item_weights.sum()

# Профілі користувачів (перші 95 — активні/warm, останні 25 — нові/cold-start)
users = {}
for u in range(NUM_USERS):
    pref_cat = CATEGORIES[u % len(CATEGORIES)]
    pref_tier = PRICE_TIERS[u % len(PRICE_TIERS)]
    is_cold = (u >= 95)
    users[f"user_{u:03d}"] = {"pref_cat": pref_cat, "pref_tier": pref_tier, "is_cold": is_cold}

# Генерація хронологічного потоку покупок (timestamp від 1 до 2000)
interactions = []
ts = 1
for u_id, u_meta in users.items():
    n_tx = 2 if u_meta["is_cold"] else random.randint(10, 18)
    for step in range(n_tx):
        # 65% покупок відповідають профілю інтересів користувача, 35% — популярним трендам
        if random.random() < 0.65:
            candidates = [iid for iid, m in items.items() if m["cat"] == u_meta["pref_cat"]]
            chosen = random.choice(candidates)
        else:
            chosen = np.random.choice(item_ids, p=item_weights)
        # Нові користувачі з'являються ближче до кінця часової шкали
        tx_time = random.randint(1450, 1950) if (u_meta["is_cold"] and step == 0) else (
            random.randint(1650, 2000) if u_meta["is_cold"] else random.randint(1, 2000)
        )
        interactions.append((tx_time, u_id, chosen))

# 2. Строге часове розбиття (Temporal Split: 80% минуле -> Train, 20% майбутнє -> Test)
interactions.sort(key=lambda x: x[0])
split_idx = int(len(interactions) * 0.80)
split_ts = interactions[split_idx][0]
train_data = interactions[:split_idx]
test_data = interactions[split_idx:]

train_user_items = {u: set() for u in users}
item_popularity = {iid: 0 for iid in item_ids}
for _, u, iid in train_data:
    train_user_items[u].add(iid)
    item_popularity[iid] += 1

test_user_items = {}
for _, u, iid in test_data:
    if iid not in train_user_items[u]:  # Оцінюємо рекомендації нових для користувача товарів
        test_user_items.setdefault(u, set()).add(iid)

print(f"Всього взаємодій: {len(interactions)} | Поріг часу split_ts = {split_ts}")
print(f"Train: {len(train_data)} транзакцій | Test: {len(test_data)} транзакцій | Тестових користувачів: {len(test_user_items)}")

# --- ПІДХІД 1: Популярнісний (Popularity-Based) ---
ranked_popular = sorted(item_ids, key=lambda x: item_popularity[x], reverse=True)

def recommend_popularity(u_id, k=10):
    seen = train_user_items[u_id]
    return [iid for iid in ranked_popular if iid not in seen][:k]

# --- ПІДХІД 2: Контентний (Content-Based за профілем та історією категорій/цін) ---
def recommend_content(u_id, k=10):
    seen = train_user_items[u_id]
    u_meta = users[u_id]
    # Профіль користувача: поєднання анкетних вподобань (для холодного старту) та частоти категорій у Train
    cat_scores = {c: (1.5 if c == u_meta["pref_cat"] else 0.0) for c in CATEGORIES}
    tier_scores = {t: (1.0 if t == u_meta["pref_tier"] else 0.0) for t in PRICE_TIERS}
    for iid in seen:
        cat_scores[items[iid]["cat"]] += 1.0
        tier_scores[items[iid]["tier"]] += 0.5

    scored = []
    for iid in item_ids:
        if iid in seen:
            continue
        score = cat_scores[items[iid]["cat"]] + 0.5 * tier_scores[items[iid]["tier"]] + 0.01 * item_popularity[iid]
        scored.append((score, iid))
    scored.sort(reverse=True)
    return [iid for _, iid in scored[:k]]

# --- ПІДХІД 3: Граф-базований (Bipartite User-Item-Category Graph + Personalized PageRank) ---
G = nx.Graph()
for u_id, u_meta in users.items():
    G.add_node(u_id, kind="user")
    # Зв'язок профілю інтересів із вузлом категорії (вирішення холодного старту в графі!)
    G.add_edge(u_id, f"cat_{u_meta['pref_cat']}", weight=1.5)

for iid, i_meta in items.items():
    G.add_node(iid, kind="item")
    G.add_edge(iid, f"cat_{i_meta['cat']}", weight=1.0)

for _, u_id, iid in train_data:
    if G.has_edge(u_id, iid):
        G[u_id][iid]["weight"] += 1.0
    else:
        G.add_edge(u_id, iid, weight=2.0)

def recommend_graph_ppr(u_id, k=10):
    seen = train_user_items[u_id]
    pers = {node: 0.0 for node in G.nodes()}
    pers[u_id] = 1.0
    ppr = nx.pagerank(G, alpha=0.85, personalization=pers, weight="weight")
    scored = [(ppr[iid], iid) for iid in item_ids if iid not in seen]
    scored.sort(reverse=True)
    return [iid for _, iid in scored[:k]]

# 3. Обчислення метрик: Precision@5, nDCG@10, Coverage, Novelty + Cold-Start аналіз
total_train_tx = len(train_data)
item_self_info = {
    iid: -math.log2((item_popularity[iid] + 1) / (total_train_tx + NUM_ITEMS))
    for iid in item_ids
}

def evaluate_recommender(name, rec_fn):
    p5_all, ndcg10_all, nov_all = [], [], []
    p5_warm, p5_cold = [], []
    ndcg_warm, ndcg_cold = [], []
    recommended_catalog = set()

    for u_id, rel_items in test_user_items.items():
        if not rel_items:
            continue
        recs = rec_fn(u_id, k=10)
        recommended_catalog.update(recs[:5])

        # Precision@5
        hits_5 = sum(1 for iid in recs[:5] if iid in rel_items)
        p5 = hits_5 / 5.0
        p5_all.append(p5)

        # nDCG@10
        dcg = sum((1.0 / math.log2(idx + 2)) for idx, iid in enumerate(recs[:10]) if iid in rel_items)
        idcg = sum((1.0 / math.log2(idx + 2)) for idx in range(min(len(rel_items), 10)))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        ndcg10_all.append(ndcg)

        # Novelty@5 (середня самоінформація рекомендованих товарів — чим вище, тим менш банальні товари)
        nov = np.mean([item_self_info[iid] for iid in recs[:5]])
        nov_all.append(nov)

        if users[u_id]["is_cold"]:
            p5_cold.append(p5)
            ndcg_cold.append(ndcg)
        else:
            p5_warm.append(p5)
            ndcg_warm.append(ndcg)

    coverage = (len(recommended_catalog) / NUM_ITEMS) * 100.0
    return {
        "name": name,
        "P@5": np.mean(p5_all),
        "nDCG@10": np.mean(ndcg10_all),
        "Coverage": coverage,
        "Novelty": np.mean(nov_all),
        "P@5_Warm": np.mean(p5_warm),
        "P@5_Cold": np.mean(p5_cold),
        "nDCG_Cold": np.mean(ndcg_cold),
    }

models = [
    evaluate_recommender("1. Popularity-Based", recommend_popularity),
    evaluate_recommender("2. Content-Based", recommend_content),
    evaluate_recommender("3. Graph-Based (PPR)", recommend_graph_ppr),
]

print("\n" + "="*105)
print(f"{'Підхід рекомендацій':<24} | {'Prec@5':<8} | {'nDCG@10':<8} | {'Coverage':<10} | {'Novelty':<8} | {'P@5 (Warm)':<10} | {'P@5 (Cold)'}")
print("="*105)
for m in models:
    print(f"{m['name']:<24} | {m['P@5']:>7.3f} | {m['nDCG@10']:>7.3f} | {m['Coverage']:>8.1f}% | {m['Novelty']:>7.2f} | {m['P@5_Warm']:>9.3f} | {m['P@5_Cold']:>9.3f}")
print("="*105)

# 4. Побудова графіка порівняння підходів та холодного старту (recsys_eval.png)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
names = ["Popularity", "Content-Based", "Graph (PPR)"]
x = np.arange(len(names))

# Графік 1: Загальна якість (Precision@5 та nDCG@10) і Warm vs Cold
ax1.bar(x - 0.2, [m["P@5"] for m in models], width=0.2, label="Precision@5 (Заг.)", color="#2563eb")
ax1.bar(x,       [m["nDCG@10"] for m in models], width=0.2, label="nDCG@10 (Заг.)", color="#0ea5e9")
ax1.bar(x + 0.2, [m["P@5_Cold"] for m in models], width=0.2, label="Precision@5 (Cold-Start)", color="#f59e0b")
ax1.set_title("Якість ранжування та стійкість до холодного старту", fontsize=11, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(names)
ax1.set_ylim(0, max(m["nDCG@10"] for m in models) * 1.35)
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.legend()

# Графік 2: Покриття каталогу (Coverage %) та Новизна (Novelty)
ax2_twin = ax2.twinx()
b1 = ax2.bar(x - 0.15, [m["Coverage"] for m in models], width=0.3, label="Catalog Coverage (%)", color="#16a34a")
b2 = ax2_twin.bar(x + 0.15, [m["Novelty"] for m in models], width=0.3, label="Novelty (bits)", color="#8b5cf6")
ax2.set_title("Різноманіття: Покриття каталогу та Новизна", fontsize=11, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(names)
ax2.set_ylabel("Coverage (%)", color="#16a34a")
ax2_twin.set_ylabel("Novelty (самоінформація)", color="#8b5cf6")
ax2.set_ylim(0, 105)
ax2.grid(axis="y", linestyle="--", alpha=0.4)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.tight_layout()
plt.savefig("recsys_eval.png", dpi=200)
print("\n[OK] Графік збережено у файл: recsys_eval.png")
