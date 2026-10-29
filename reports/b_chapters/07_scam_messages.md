# Chapter 7. Scam messages and Naive Bayes

> Real transaction data is legally unavailable; we built a simulator calibrated to NPCI/RBI statistics and validated against the IBM AML benchmark. The scam-message dataset is the one fully real dataset in the project.

> **Status: preliminary.** The Google Form is collecting messages. Until `messages.csv` exists, the pipeline runs on ~36 synthetic messages we wrote ourselves, and the numbers in [`b_07_scam_text.md`](../b_07_scam_text.md) only show that it works. Rerun `python -m text.classifier` when the data arrives and update section 7.4.

## 7.1 Collection and ethics

Victims reach mule networks through scam messages: fake KYC alerts, "digital arrest" calls, task-based job offers, loan apps and fake investment groups. We collect real examples with a Google Form: message text, where it arrived (SMS / WhatsApp / other), scam type (or "not sure" / "genuine"), language, and a consent tick-box. The form asks contributors to delete names, numbers, links, UPI IDs, OTPs and account numbers before pasting.

Two more layers protect privacy:
1. **Automatic scrubbing** (`text/anonymise.py`) replaces phone numbers, URLs, UPI IDs, e-mails, OTPs and account/card numbers with `<PHONE>`, `<URL>`, `<UPI>`, `<EMAIL>`, `<OTP>`, `<ACCOUNT>`, while keeping rupee amounts. It flags any message with a likely name ("Dear Ramesh") or leftover digits.
2. **Human review** of every flagged row before `messages.csv` is committed. The raw form export never enters git.

Rows without consent are dropped, and duplicates are removed on normalised text.

## 7.2 Naive Bayes

By Bayes' rule, `P(class | message) ∝ P(class) · P(message | class)`. The *naive* assumption treats tokens as independent given the class, so `log P(class | message) = log P(class) + Σ count(t) · log P(t | class) + const`. It is clearly false for n-grams, which overlap, but it mainly makes the probabilities over-confident; the *ranking* of classes is usually still right, and with few messages the small number of parameters is a strength. Laplace smoothing, `P(t | class) = (count + α) / (total + αV)`, stops one unseen token from zeroing a class; α ∈ {0.1, 0.3, 1.0} is chosen by cross-validation.

Character n-grams (`char_wb`, 2-5) handle Hinglish spelling ("aapka", "apka", "apke" share "apk", "pka"). The placeholder tokens are kept as features: a link or UPI id in a message is itself evidence.

## 7.3 Models and evaluation

Four pipelines: (a) MultinomialNB on word 1-2-grams, (b) on char 2-5-grams, (c) on both, (d) TF-IDF logistic regression (`class_weight="balanced"`) as a non-Bayesian baseline. A stratified 80/20 hold-out by scam type (seed 0), CV on the 80% for α and C, macro-F1 as the metric (accuracy would reward always saying "scam"). Two tasks: binary (scam vs genuine) and multiclass (scam type, classes under 5 messages merged into `other_scam`). We also report a confusion matrix by scam type, a slice by language, the top scam-indicating words and character n-grams (`log P(t | scam) − log P(t | genuine)`), and a worked per-token example.

## 7.4 Results

*To be filled from `reports/b_tables/text_*.csv` once the real data is in.* On the synthetic fixtures, macro-F1 is around 0.4-0.5, which with 7 test messages says nothing; the top tokens (`<url>`, `kyc`, `arrest`, `loan`) show the features behave as intended.

## 7.5 Limitations

- **Small n:** a few hundred messages at best, so hold-out numbers have wide uncertainty.
- **Self-selection:** contributors pick memorable, blatant scams, and few send genuine messages.
- **Label noise:** scam types are the contributors' own judgement.
- **Scrubbing hides signal:** `<URL>` hides which domain a link points to.
- If fewer than 300 own messages arrive, the UCI SMS Spam Collection may supplement training (B-D17, licence to be checked), always evaluated on own messages only.
