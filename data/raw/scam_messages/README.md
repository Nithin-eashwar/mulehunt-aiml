# Scam-message dataset

The one fully real dataset in MuleHunt: Indian scam (and genuine) messages collected by
Person B through a Google Form, with consent, then anonymised and reviewed by hand.

## Files

| File | What it is |
|---|---|
| `messages.csv` | the real, anonymised, reviewed messages. **Not committed yet**: collection is in progress. |
| `sample_messages.csv` | ~36 messages **written by us (synthetic)** as test fixtures and demo examples. Never used in reported results. Placeholder tokens (`<URL>`, `<PHONE>`...) mimic the anonymiser's output. |

Schema (both files): `msg_id, text, label, scam_type, language, channel, source`
- `label` in {scam, genuine}; `scam_type` in {fake_kyc, digital_arrest, job_task, loan_app, fake_investment, other_scam, genuine}
- `language` in {english, hindi, hinglish, other}; `channel` in {sms, whatsapp, other}; `source` in {form, self, public}

## Collection method

1. Google Form fields: message text, where it was received (SMS / WhatsApp / other), scam type
   (the list above plus "not sure" and "genuine"), language, and a consent tick-box ("I agree
   this anonymised text may be used in a student project"). The form tells contributors to delete
   names, phone numbers, links, UPI IDs, OTPs and account numbers before pasting.
2. `python -m text.dataset import <form_export.csv>` keeps only consenting rows, anonymises
   (`text/anonymise.py`), de-duplicates and writes `messages.csv`, printing rows that need review.
3. A person reviews every flagged row (and skims the rest) before `messages.csv` is committed.
   **The raw form export never goes in git.**

## Public supplement (B-D17)

Only if fewer than 300 own messages: the UCI *SMS Spam Collection* (tag `source = public`),
evaluated on own messages only. Check its licence before use. Not used so far.

## Counts

Filled in by `python -m text.classifier` in `reports/b_07_scam_text.md`.
