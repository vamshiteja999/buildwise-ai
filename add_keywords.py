import csv, sys
from pathlib import Path

path = Path("golden_questions.csv")
KW = [
 "inspection|investigation", "subpart l|subpart cc|subpart r", "2 inches", "4 in 12", "39 inches",
 "6 feet|guardrail", "infeasible|1926.502(k)", "personal fall arrest", "6 feet|cover", "6 feet|barricade",
 "10 inches", "50 feet|safety monitoring", "hard hat|toeboard", "(b)(15)", "39 inches|6 feet",
 "42 inches", "200 pounds|150 pounds", "21 inches|19 inches", "8 feet|13 feet", "400|sand",
 "1,800|3.5 feet", "5,000 pounds", "not acceptable", "2 feet|3,000", "6 feet|10 feet",
 "34 inches|39 inches|45 inches", "6 feet|25 feet|60 feet", "competent person|monitor", "twice|cover",
 "toeboard|50 pounds", "qualified person|controlled access zone", "rescue", "competent person|fall hazards",
 "signature|date", "obsolete", "291|828", "3,000 pounds", "wood frame|dwelling", "1926.501(b)(7)(ii)",
 "32 inches", "36 in|44 in", "300 pounds|19 inches", "60 inches|19 inches", "34 inches|25 percent",
 "60 inches", "a, b", "a, b", "design pressure", "asce 7-22|frsa", "drainage|calculations",
 "eighth|ninth", "roofing|plumbing|gas", "notice of commencement", "$5,000|owner", "bonding|mortgage",
 "standing seam", "wellington|a3.01", "core construction|18", "7,300|apparatus", "currie sowards aguila",
 "220604", "05 12 05|shop drawings", "cmu|eifs", "1979|summer", "mcclellan|facilities", "eighth",
 "", "", "", "", "",
]
rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
if len(rows) != len(KW):
    sys.exit(f"Expected {len(KW)} rows but found {len(rows)}. Use golden_questions_v2.csv instead.")
if "answer_keywords" in rows[0]:
    sys.exit("answer_keywords already present. Nothing to do.")
for row, kw in zip(rows, KW):
    row["answer_keywords"] = kw
    row["expected_source"] = row["expected_source"].replace("Supplemental_Roof", "Supplemental Roof")
cols = ["question", "expected_source", "expected_answer", "type", "answer_keywords", "expected_section", "focus"]
with path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
print(f"Updated {len(rows)} rows; {sum(1 for k in KW if k)} have keywords.")