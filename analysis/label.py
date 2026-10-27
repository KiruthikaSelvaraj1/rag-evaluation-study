import csv
A = set("128 1362 1363 820 821 830 831 913 914 715 716 1175 1179".split())
B = set("1 13 132 239 303 312 314 421 431 475 508 551 560 619 768 975 1100 1110 1199 1213 1226 1272 1316 1332".split())
C = set("324 437 502 517 660 690 834 887 1191".split())
rows = list(csv.DictReader(open("analysis/errors_vector.csv", encoding="utf-8")))
for r in rows:
    r["category"] = ("A_sibling_claims" if r["qid"] in A else
                     "B_right_topic_wrong_paper" if r["qid"] in B else
                     "C_gold_not_visible" if r["qid"] in C else "")
w = csv.DictWriter(open("analysis/errors_vector_labeled.csv", "w", newline="", encoding="utf-8"), fieldnames=rows[0].keys())
w.writeheader(); w.writerows(rows)
print({k: sum(r["category"] == k for r in rows) for k in {r["category"] for r in rows}})