import csv, sys
from rag_study.judge import cohens_kappa

def load(p):
    return {(r["system"], r["qid"]): int(r["human_faithful"])
            for r in csv.DictReader(open(p, encoding="utf-8-sig")) if r["human_faithful"] in ("0", "1")}

a, b = load(sys.argv[1]), load(sys.argv[2])
keys = sorted(set(a) & set(b))
x, y = [a[k] for k in keys], [b[k] for k in keys]
print("n =", len(keys), " agreement =", round(sum(i == j for i, j in zip(x, y)) / len(keys), 3),
      " kappa =", round(cohens_kappa(x, y), 3))
print("round 1 faithful:", sum(x), "/", len(x), "   round 2 faithful:", sum(y), "/", len(y))