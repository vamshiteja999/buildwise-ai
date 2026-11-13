"""Are the 3 changed answers noise or a regression? Runs each question 3 times and shows what happened.
Run with the app closed:  python check_flips.py
"""
import time
from config import VERIFY, TEMPERATURE, COLLECTION
from buildwise.core import answer_question, LAST_VERIFY, REFUSAL_TEXT

QUESTIONS = [
    ("job number / drawing number (Run E: answered)", "What job number and drawing number does the Station 25 elevations sheet show?"),
    ("basic wind speed (Run E: refused, correct)", "What basic wind speed applies to roofing design in Boca Raton?"),
    ("fall protection plan (Run E: all keywords)", "Who may use a fall protection plan and what must the plan include?"),
]
print(f"VERIFY={VERIFY} TEMPERATURE={TEMPERATURE} COLLECTION={COLLECTION}\n")
for label, q in QUESTIONS:
    print("==", label)
    for i in range(3):
        LAST_VERIFY.clear()
        t0 = time.time()
        answer, _ = answer_question(q)
        refused = answer.strip() == REFUSAL_TEXT
        print(f"  run {i+1}: {'REFUSED' if refused else 'answered'}  {time.time()-t0:4.1f}s  verify={dict(LAST_VERIFY) or 'not run'}  | {answer[:90]!r}")
