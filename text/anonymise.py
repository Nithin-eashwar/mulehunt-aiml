"""Scrub personal data from scam messages before they enter the repo (Task 9).

`anonymise(text) -> (clean_text, needs_review)`. Replaces phone numbers, URLs, UPI ids,
e-mails, OTP-like codes and account/card numbers with placeholder tokens; keeps rupee
amounts. `needs_review` flags a likely name after a salutation, or a run of 4+ digits
left outside a rupee amount. Automatic scrubbing never replaces the human review.
"""

import re

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
UPI = re.compile(r"\b[\w.-]+@[A-Za-z][A-Za-z0-9]*\b")
URL = re.compile(r"(?:https?://|www\.)\S+"
                 r"|\b(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|rb\.gy|cutt\.ly|is\.gd|t\.ly|wa\.me)/\S*"
                 r"|\b[\w-]+(?:\.[\w-]+)*\.(?:com|in|net|org|co|io|xyz|info|top|link|app|online|site|club)\b(?:/\S*)?",
                 re.I)
PHONE = re.compile(r"(?<![\w<])(?:\+91[\s-]?|91[\s-]|0)?[6-9](?:[\s-]?\d){9}(?!\d)")
OTP_AFTER = re.compile(r"\b(OTP|code|PIN)\b(\D{0,15}?)(?<!\d)\d{4,8}(?!\d)", re.I)
OTP_BEFORE = re.compile(r"(?<!\d)\d{4,8}(?!\d)(?=\s*(?:is\s+(?:your|the)\s+)?(?:OTP|code|PIN)\b)", re.I)
ACCOUNT = re.compile(r"(?<![\d<])\d(?:[ -]?\d){8,17}(?!\d)")
AMOUNT = re.compile(r"(?:₹|\bRs\.?|\bINR)\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:rs\b|rupees\b|/-)", re.I)
NAME = re.compile(r"\b(?:Dear|Mr|Mrs|Ms|Shri|Smt|Sir|Madam)\.?\s+(?!Customer\b|User\b|Sir\b|Madam\b|Member\b)"
                  r"[A-Z][a-z]+")
DIGITS = re.compile(r"\d{4,}")


def anonymise(text: str) -> tuple[str, bool]:
    t = str(text)
    t = EMAIL.sub("<EMAIL>", t)
    t = UPI.sub("<UPI>", t)
    t = URL.sub("<URL>", t)
    t = OTP_AFTER.sub(lambda m: f"{m.group(1)}{m.group(2)}<OTP>", t)
    t = OTP_BEFORE.sub("<OTP>", t)
    # protect rupee amounts from the phone / account patterns
    amounts = []
    t = AMOUNT.sub(lambda m: amounts.append(m.group(0)) or f"\x00{len(amounts) - 1}\x00", t)
    t = PHONE.sub("<PHONE>", t)
    t = ACCOUNT.sub("<ACCOUNT>", t)
    leftover = bool(DIGITS.search(re.sub(r"\x00\d+\x00", "", t)))
    t = re.sub(r"\x00(\d+)\x00", lambda m: amounts[int(m.group(1))], t)
    return t, bool(NAME.search(t)) or leftover
