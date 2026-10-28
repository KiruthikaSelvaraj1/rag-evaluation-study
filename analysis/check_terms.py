import csv, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
system, qid, *terms = sys.argv[1:]
rows = [r for r in csv.DictReader(open("data/human_labels/labels.csv", encoding="utf-8"))
        if r["system"] == system and r["qid"] == qid]
p = rows[0]["passages"].lower()
for t in terms:
    print(f"{t!r:28} {'FOUND' if t.lower() in p else 'MISSING'}")