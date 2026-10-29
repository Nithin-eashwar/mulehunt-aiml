# Chapter 7. Scam messages and Naive Bayes

> Real transaction data is legally unavailable; we built a simulator calibrated to NPCI/RBI statistics and validated against the IBM AML benchmark.

Victims reach mule networks through scam messages: fake KYC alerts, "digital arrest" calls, task-based job offers, loan apps and fake investment groups. This chapter builds a classifier that reads such a message and says whether it is a scam, and which kind. The full tables are in [`b_07_scam_text.md`](../b_07_scam_text.md) (`python -m text.build_corpus`, then `python -m text.classifier`); the numbers here come from `reports/b_tables/text_*.csv`.

## 7.1 The data, and what is real

We could not collect enough messages from people to train on, so the corpus has three clearly separated parts. **No message was collected from a person.**

| Source | What it is | Real? | Scam | Genuine |
|---|---|---|---|---|
| `public` | UCI SMS Spam Collection (Almeida & Hidalgo 2011, CC BY 4.0): English SMS from the UK and Singapore, ~2011 | real, not Indian | 608 | 4,501 |
| `reported` | written by us from scam messages described in 26 cited Indian sources: PIB and DoT notices, bank advisories, police warnings, fact-checkers and news reports. Each message keeps its source URL | real patterns, our wording | 56 | 0 |
| `synthetic` | generated from seeded templates (`text/synthetic.py`): the five Indian scam types plus other scams, and ordinary Indian messages (bank alerts, OTPs, deliveries, family chat), in English, Hinglish and Hindi | synthetic | 360 | 200 |

The synthetic genuine messages matter. Without them, "Indian-sounding" would itself predict "scam": every Indian message would be a scam and every genuine message British. With them, genuine bank alerts that say "KYC", "UPI" or "Rs" appear on both sides.

All messages pass through the anonymiser (`text/anonymise.py`), so phone numbers, links, UPI IDs, OTPs and account numbers become `<PHONE>`, `<URL>`, `<UPI>`, `<OTP>` and `<ACCOUNT>` in every source. The Google Form import tool is kept for any messages collected later (`source = form`).

## 7.2 Naive Bayes

By Bayes' rule, `P(class | message) ∝ P(class) · P(message | class)`. The *naive* assumption treats tokens as independent given the class, so `log P(class | message) = log P(class) + Σ count(t) · log P(t | class) + const`. It is clearly false for n-grams, which overlap, but it mainly makes the probabilities over-confident. The *ranking* of classes is usually still right, and the small number of parameters suits small data. Laplace smoothing, `P(t | class) = (count + α) / (total + αV)`, stops one unseen token from zeroing a class; α ∈ {0.1, 0.3, 1.0} is chosen by cross-validation.

Character n-grams (`char_wb`, 2-5) handle Hinglish spelling ("aapka", "apka", "apke" share "apk" and "pka"). The placeholder tokens are kept as features, because a link or a phone number in a message is itself evidence: `<phone>` is the single most scam-indicating word feature (log-ratio 6.5).

## 7.3 How we evaluate

A random 80/20 split would flatter the model. Messages generated from the same template land on both sides, so the test set is close to the training set. Our **headline is a transfer test** instead:

- **Train:** the synthetic Indian messages plus 80% of UCI.
- **Test:** all 56 reported messages, which were never in training and were written from real cases rather than templates, plus the held-out 20% of UCI.

The four models are (a) MultinomialNB on word 1-2-grams, (b) MultinomialNB on character 2-5-grams, (c) both, and (d) a TF-IDF logistic regression baseline. Each picks its α or C by stratified 5-fold CV on its training part only, then is evaluated once.

## 7.4 Results

**Scam vs genuine, transfer test** (`text_transfer_binary.csv`):

| Model | Test macro-F1 |
|---|---|
| NB, words | 0.966 |
| NB, characters | 0.974 |
| NB, words + characters | 0.976 |
| TF-IDF logistic regression | **0.978** |

The best model flags **54 of the 56** reported Indian scam messages (recall 0.964). It wrongly flags 0.4% of the 900 held-out genuine SMS. By type, recall is 1.0 for `fake_kyc`, `job_task`, `loan_app` and other scams, 0.9 for `digital_arrest` and 0.83 for `fake_investment`. The misses are a "digital arrest" message with no link, number or amount, and an investment pitch that reads like a friendly invitation. Both are scams that work by sounding ordinary.

**Which scam type** (trained on synthetic, tested on the reported messages): the word Naive Bayes model reaches macro-F1 **0.960**, ahead of logistic regression (0.837). Naive Bayes wins this smaller, cleaner task.

**For comparison**, the random split gives 0.987 (binary) and 0.995 (type). That this beats the transfer test is the expected template leak, which is why it is not the headline.

## 7.5 Limitations

- **No collected messages.** The Indian part is written by us, from cited real cases (reported) or from templates (synthetic). Real messages are messier, with typos, mixed scripts and odd formatting.
- **One author.** The reported and synthetic messages share a writer, so even the transfer test is easier than truly unseen messages would be. These numbers are an upper bound.
- **UCI is old and not Indian.** It contributes real genuine SMS and generic spam from about 2011, which we label `other_scam`.
- **Every genuine message in the test set is a UK or Singapore SMS.** So "0.4% false alarms" says nothing about genuine Indian bank SMS (OTPs, debit alerts, KYC-complete notices), which look a lot like scams. The cheap fix is a held-out set of 30-50 anonymised genuine bank and UPI SMS from our own phones; until then, treat the false-alarm rate as untested on Indian messages.
- **56 reported messages:** one message changes a type's recall by 10-17 points.
- **Placeholders hide signal**, such as which domain a link points to.
