"""Terminal labelling tool. Shows one answer at a time; type 1, 0, s (skip), b (back) or q (quit).
Reads data/human_labels/to_label.csv, saves progress to data/human_labels/labels.csv after every row."""
import csv, os, sys, textwrap

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SRC, DST = "data/human_labels/to_label.csv", "data/human_labels/labels.csv"

rows = list(csv.DictReader(open(DST if os.path.exists(DST) else SRC, encoding="utf-8-sig")))
fields = list(rows[0].keys())

def save():
    with open(DST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

def show(i, r):
    print("\n" + "=" * 78)
    print(f"Row {i + 1}/{len(rows)}   labelled so far: {sum(x['human_faithful'] in ('0', '1') for x in rows)}")
    print("=" * 78)
    print("CLAIM/QUESTION:", r["query"])
    print("\nANSWER TO CHECK:\n" + textwrap.fill(r["answer"], 100))
    print("\nPASSAGES (the only evidence you may use):")
    for line in r["passages"].split("\n"):
        print(textwrap.fill(line, 100, subsequent_indent="    "), "\n")
    print("Is EVERY claim in the answer supported by these passages?")

i = next((k for k, r in enumerate(rows) if r["human_faithful"] not in ("0", "1")), 0)
while 0 <= i < len(rows):
    show(i, rows[i])
    a = input("[1 = yes, all supported | 0 = no | s = skip | b = back | q = quit] > ").strip().lower()
    if a in ("0", "1"):
        rows[i]["human_faithful"] = a; save(); i += 1
    elif a == "s":
        i += 1
    elif a == "b":
        i = max(0, i - 1)
    elif a == "q":
        break
save()
print(f"\nSaved {sum(x['human_faithful'] in ('0', '1') for x in rows)} labels to {DST}")
