"""Naive Bayes scam-message classifier (Task 10).

    python -m text.classifier [--messages data/raw/scam_messages/messages.csv]

Four scikit-learn pipelines (word NB, char NB, word+char NB, TF-IDF logistic regression),
binary (scam vs genuine) and multiclass (scam type) tasks, a stratified 80/20 hold-out
plus CV on the 80% for alpha / C. Saves the best pipelines to text/artifacts/ and writes
reports/b_07_scam_text.md. Without messages.csv it runs on the synthetic fixtures and
marks the report PRELIMINARY.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

from mulehunt.b_params import HONEST_FRAMING, TEXT_MODEL_PATH, params_table
from text.dataset import load_messages

REPORT = Path("reports/b_07_scam_text.md")
FIGURES = Path("reports/b_figures")
TABLES = Path("reports/b_tables")
TOKEN_PATTERN = r"(?u)<\w+>|\b\w\w+\b"  # keeps <url>, <phone>, <upi> ... as single tokens
MIN_CLASS = 5
SEED = 0
NB_ALPHAS = (0.1, 0.3, 1.0)
LR_CS = (0.3, 1.0, 3.0, 10.0)


def _word(cls=CountVectorizer):
    return cls(lowercase=True, ngram_range=(1, 2), token_pattern=TOKEN_PATTERN)


def _char(cls=CountVectorizer):
    return cls(lowercase=True, analyzer="char_wb", ngram_range=(2, 5))


def build_models() -> dict[str, tuple[Pipeline, dict]]:
    """name -> (pipeline, parameter grid)."""
    nb = {"model__alpha": list(NB_ALPHAS)}
    return {
        "nb_word": (Pipeline([("features", _word()), ("model", MultinomialNB())]), nb),
        "nb_char": (Pipeline([("features", _char()), ("model", MultinomialNB())]), nb),
        "nb_word_char": (Pipeline([("features", FeatureUnion([("word", _word()), ("char", _char())])),
                                   ("model", MultinomialNB())]), nb),
        "lr_tfidf": (Pipeline([("features", FeatureUnion([("word", _word(TfidfVectorizer)),
                                                          ("char", _char(TfidfVectorizer))])),
                               ("model", LogisticRegression(class_weight="balanced", max_iter=2000))]),
                     {"model__C": list(LR_CS)}),
    }


def multiclass_labels(df: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    counts = df.scam_type.value_counts()
    small = [t for t, n in counts.items() if n < MIN_CLASS and t not in ("other_scam", "genuine")]
    return df.scam_type.where(~df.scam_type.isin(small), "other_scam"), small


def _cv(y):
    n = int(min(5, pd.Series(y).value_counts().min()))
    return StratifiedKFold(n_splits=max(n, 2), shuffle=True, random_state=SEED)


def fit_task(texts, y, strat) -> dict:
    """Hold-out + CV for every model; returns per-model results and the fitted searches."""
    Xtr, Xte, ytr, yte = train_test_split(texts, y, test_size=0.2, random_state=SEED, stratify=strat)
    out = {}
    for name, (pipe, grid) in build_models().items():
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=_cv(ytr))
        search.fit(Xtr, ytr)
        pred = search.predict(Xte)
        out[name] = {"search": search, "cv_macro_f1": search.best_score_, "best_params": search.best_params_,
                     "test_macro_f1": f1_score(yte, pred, average="macro"), "pred": pred}
    return {"models": out, "Xte": Xte, "yte": yte, "Xtr": Xtr, "ytr": ytr}


def _weights(pipe) -> np.ndarray:
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_log_prob_"):
        return model.feature_log_prob_[1] - model.feature_log_prob_[0]
    return model.coef_[0]


def top_features(pipe, k=20, prefix=None) -> pd.DataFrame:
    names = pipe.named_steps["features"].get_feature_names_out()
    w = _weights(pipe)
    order = [i for i in np.argsort(-w) if prefix is None or names[i].startswith(prefix)][:k]
    return pd.DataFrame({"feature": [names[i] for i in order], "log_ratio_scam_vs_genuine": np.round(w[order], 3)})


def token_contributions(pipe, text: str, k=5) -> list[tuple[str, float]]:
    X = pipe.named_steps["features"].transform([text])
    names = pipe.named_steps["features"].get_feature_names_out()
    w = _weights(pipe)
    idx = X.nonzero()[1]
    contrib = sorted(((names[i], float(w[i] * X[0, i])) for i in idx), key=lambda t: -t[1])
    return [(n.split("__")[-1], round(c, 3)) for n, c in contrib[:k]]


def worked_example(pipe, text: str) -> str:
    """Per-token log-probabilities of the word NB model for one message."""
    vec, nb = pipe.named_steps["features"], pipe.named_steps["model"]
    X = vec.transform([text])
    names = vec.get_feature_names_out()
    rows = [f"Message: *{text}*", "", "| token | count | log P(t \\| scam) | log P(t \\| genuine) | difference |",
            "|---|---|---|---|---|"]
    total = 0.0
    for i in X.nonzero()[1]:
        if " " in names[i]:
            continue  # unigrams only, to keep the example readable
        c = X[0, i]
        ls, lg = nb.feature_log_prob_[1, i], nb.feature_log_prob_[0, i]
        total += c * (ls - lg)
        rows.append(f"| `{names[i]}` | {c} | {ls:.3f} | {lg:.3f} | {ls - lg:+.3f} |")
    prior = nb.class_log_prior_[1] - nb.class_log_prior_[0]
    full = float(pipe.predict_log_proba([text])[0, 1] - pipe.predict_log_proba([text])[0, 0])
    rows += ["", f"Prior log-odds log P(scam) - log P(genuine) = {prior:+.3f}; the unigram terms add {total:+.3f}; "
             f"with the bigrams the model's posterior log-odds are {full:+.3f}, i.e. P(scam | message) = "
             f"{1 / (1 + math.exp(-full)):.3f}."]
    return "\n".join(rows)


# ---------------------------------------------------------------- saved model

def save_artifact(binary, multiclass, labels, n, path=TEXT_MODEL_PATH):
    import joblib
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"binary": binary, "multiclass": multiclass, "labels": labels, "trained_on": n}, path, compress=3)


def load_artifact(path=TEXT_MODEL_PATH):
    import joblib
    return joblib.load(path) if Path(path).exists() else None


def predict_message(text: str, artifact=None) -> dict:
    art = artifact or load_artifact()
    if art is None:
        raise FileNotFoundError(f"{TEXT_MODEL_PATH} missing; run python -m text.classifier")
    p = float(art["binary"].predict_proba([text])[0, 1])
    return {"scam_prob": p, "scam_type": str(art["multiclass"].predict([text])[0]),
            "top_tokens": token_contributions(art["binary"], text)}


# ---------------------------------------------------------------- study

def _confusion_figure(y, pred, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Scam type: confusion matrix (hold-out)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _results_table(task) -> pd.DataFrame:
    return pd.DataFrame([{"model": n, "best_params": str(r["best_params"]), "cv_macro_f1": round(r["cv_macro_f1"], 3),
                          "holdout_macro_f1": round(r["test_macro_f1"], 3)} for n, r in task["models"].items()])


def _best(task) -> str:
    return max(task["models"], key=lambda n: (task["models"][n]["cv_macro_f1"], n.startswith("nb")))


THEORY = """
**Bayes' rule.** For a message with tokens `t_1 ... t_n`,
`P(class | message) = P(message | class) P(class) / P(message)`; the classifier picks the class with
the largest numerator.

**The naive assumption.** Tokens are treated as independent given the class, so
`P(message | class) = prod_i P(t_i | class)`, and in logs the score is
`log P(class) + sum_i count_i x log P(t_i | class)`. MultinomialNB estimates `P(t | class)` from token
counts in the training messages of that class.

**Why it is false yet works.** n-grams overlap ("kyc", "kyc update", and the character n-grams
"ky", "kyc", "yc u" all describe the same words), so tokens are clearly dependent. Independence
mainly makes the probabilities over-confident; the *ranking* of classes, which is all a decision
needs, is often still right. With few messages and many features, the strong assumption is also a
useful bias: there are far fewer parameters to estimate than in a model of token interactions.

**Laplace / `alpha` smoothing.** `P(t | class) = (count(t, class) + alpha) / (total(class) + alpha x V)`.
Without it, one token never seen in a class would make that class's probability zero. `alpha` is
chosen by cross-validation from {0.1, 0.3, 1.0}.

**Why character n-grams.** Hinglish has no fixed spelling ("aapka", "apka", "apke"); `char_wb`
2-5-grams share most of their pieces ("apk", "pka"), so the variants support each other. The
anonymiser's placeholders (`<url>`, `<phone>`, `<upi>`) are kept as tokens: a link or a UPI id in a
message is itself evidence.
"""


def fit_on(Xtr, ytr, Xte, yte) -> dict:
    """CV model selection on the training set only, then one evaluation on the given test set."""
    out = {}
    for name, (pipe, grid) in build_models().items():
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=_cv(ytr))
        search.fit(Xtr, ytr)
        pred = search.predict(Xte)
        out[name] = {"search": search, "cv_macro_f1": search.best_score_, "best_params": search.best_params_,
                     "test_macro_f1": f1_score(yte, pred, average="macro", zero_division=0), "pred": pred}
    return {"models": out, "Xte": list(Xte), "yte": np.asarray(yte), "Xtr": list(Xtr), "ytr": np.asarray(ytr)}


def _count(df, col):
    return df[col].value_counts().rename("messages").rename_axis("value").reset_index().assign(by=col)


def run_text_study(messages=None, out=REPORT, fig_dir=FIGURES, tab_dir=TABLES, model_path=TEXT_MODEL_PATH):
    from mulehunt.report import write_markdown_report
    fig_dir, tab_dir = Path(fig_dir), Path(tab_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    df = load_messages(messages).reset_index(drop=True)
    texts = df.text.tolist()
    y_bin = (df.label == "scam").astype(int).to_numpy()
    y_multi, merged = multiclass_labels(df)
    indian = df.source.isin(["reported", "synthetic", "form", "self"])

    by_source = df.groupby(["source", "label"]).size().unstack(fill_value=0).reset_index()
    by_source.to_csv(tab_dir / "text_by_source.csv", index=False)
    counts = pd.concat([_count(df, "scam_type"), _count(df, "language"), _count(df, "channel"),
                        _count(df, "source")])[["by", "value", "messages"]]
    counts.to_csv(tab_dir / "text_counts.csv", index=False)

    # ---- headline: transfer test. Train on synthetic + 80% of UCI; test on the cited reported
    # messages (never seen in training) + the held-out 20% of UCI.
    transfer = None
    if (df.source == "reported").any():
        uci = df[df.source == "public"]
        uci_tr, uci_te = train_test_split(uci.index, test_size=0.2, random_state=SEED, stratify=uci.label)
        tr_idx = list(df.index[df.source == "synthetic"]) + list(uci_tr)
        te_idx = list(df.index[df.source == "reported"]) + list(uci_te)
        transfer = fit_on(df.text[tr_idx], y_bin[tr_idx], df.text[te_idx], y_bin[te_idx])
        t_name = _best(transfer)
        pred = pd.Series(transfer["models"][t_name]["pred"], index=te_idx)
        rep = df.loc[df.source == "reported"]
        rep_recall = rep.assign(caught=pred[rep.index].astype(bool)).groupby("scam_type").caught.agg(
            ["size", "sum", "mean"]).rename(columns={"size": "messages", "sum": "flagged", "mean": "recall"}).round(3)
        rep_recall.loc["all"] = [len(rep), int(pred[rep.index].sum()), round(pred[rep.index].mean(), 3)]
        rep_recall = rep_recall.reset_index()
        ham_te = [i for i in uci_te if y_bin[i] == 0]
        spam_te = [i for i in uci_te if y_bin[i] == 1]
        transfer_tab = _results_table(transfer)
        transfer_tab.to_csv(tab_dir / "text_transfer_binary.csv", index=False)
        rep_recall.to_csv(tab_dir / "text_transfer_reported_recall.csv", index=False)
        t_fpr = float(pred[ham_te].mean())
        t_spam = float(pred[spam_te].mean())

        # multiclass transfer on Indian scam types: train synthetic, test reported
        syn = df[df.source == "synthetic"]
        mt = fit_on(syn.text, syn.scam_type, rep.text, rep.scam_type)
        mt_name = _best(mt)
        mt_labels = sorted(set(syn.scam_type) | set(rep.scam_type))
        mt_tab = _results_table(mt)
        mt_tab.to_csv(tab_dir / "text_transfer_multiclass.csv", index=False)
        mt_per = pd.DataFrame(classification_report(mt["yte"], mt["models"][mt_name]["pred"], labels=mt_labels,
                                                    output_dict=True, zero_division=0)).T.round(3)
        mt_per = mt_per.reset_index(names="class")
        mt_per.to_csv(tab_dir / "text_transfer_per_class.csv", index=False)
        f_cm = _confusion_figure(mt["yte"], mt["models"][mt_name]["pred"], mt_labels, fig_dir / "text_confusion.png")
        transfer = {"t_name": t_name, "tab": transfer_tab, "rep_recall": rep_recall, "fpr": t_fpr,
                    "uci_spam_recall": t_spam, "mt_name": mt_name, "mt_tab": mt_tab, "mt_per": mt_per, "f_cm": f_cm,
                    "macro": transfer["models"][t_name]["test_macro_f1"],
                    "mt_macro": mt["models"][mt_name]["test_macro_f1"], "n_rep": len(rep), "n_ham": len(ham_te)}

    # ---- comparison: an ordinary stratified 80/20 split of everything (optimistic: templates leak)
    binary = fit_task(texts, y_bin, df.scam_type)
    bin_tab = _results_table(binary)
    bin_tab.to_csv(tab_dir / "text_binary.csv", index=False)
    b_name = _best(binary)
    multi = fit_task(texts, y_multi.to_numpy(), y_multi)
    multi_tab = _results_table(multi)
    multi_tab.to_csv(tab_dir / "text_multiclass.csv", index=False)
    m_name = _best(multi)
    labels = sorted(set(y_multi))
    if transfer is None:
        transfer_f_cm = _confusion_figure(multi["yte"], multi["models"][m_name]["pred"], labels,
                                          fig_dir / "text_confusion.png")

    # language slice (random split, binary)
    hold = pd.DataFrame({"text": binary["Xte"], "y": binary["yte"], "pred": binary["models"][b_name]["pred"]})
    hold["language"] = hold.text.map(dict(zip(df.text, df.language)))
    hold["indian"] = hold.text.map(dict(zip(df.text, indian)))
    rows = []
    for (lg, ind), g in hold.groupby(["language", "indian"]):
        rows.append({"language": lg, "subset": "Indian" if ind else "UCI", "messages": len(g),
                     "accuracy": round((g.y == g.pred).mean(), 3),
                     "macro_f1": round(f1_score(g.y, g.pred, average="macro", zero_division=0), 3)})
    lang_tab = pd.DataFrame(rows)
    lang_tab.to_csv(tab_dir / "text_by_language.csv", index=False)

    # top features (NB on all messages)
    word_nb = build_models()["nb_word"][0].set_params(model__alpha=1.0).fit(texts, y_bin)
    char_nb = build_models()["nb_char"][0].set_params(model__alpha=1.0).fit(texts, y_bin)
    top_w, top_c = top_features(word_nb), top_features(char_nb)
    top_w.to_csv(tab_dir / "text_top_words.csv", index=False)
    top_c.to_csv(tab_dir / "text_top_chars.csv", index=False)
    example = (df[df.source == "reported"] if (df.source == "reported").any() else df[df.label == "scam"]).text.iloc[0]

    # saved model: binary on everything; multiclass on the Indian messages (UCI spam is not an Indian type)
    bin_pipe = binary["models"][b_name]["search"].best_estimator_.fit(texts, y_bin)
    ind = df[indian] if indian.any() else df
    multi_pipe = multi["models"][m_name]["search"].best_estimator_.fit(ind.text, ind.scam_type)
    save_artifact(bin_pipe, multi_pipe, sorted(set(ind.scam_type)), len(df), model_path)

    synthetic_only = df.attrs["synthetic"]
    banner = ("**PRELIMINARY: synthetic fixtures only.** Run `python -m text.build_corpus` first.\n\n"
              if synthetic_only else "")
    if transfer:
        t = transfer
        rr = t["rep_recall"].set_index("scam_type")
        headline = f"""
- **Transfer test (headline).** Trained on synthetic Indian messages + 80% of UCI, tested on the {t['n_rep']}
  reported Indian scam messages (never in training) + the held-out 20% of UCI. Best binary model
  `{t['t_name']}`: macro-F1 {t['macro']:.3f}; it flags **{rr.loc['all', 'flagged']:.0f} of {t['n_rep']}** reported
  scams (recall {rr.loc['all', 'recall']:.3f}) and wrongly flags {t['fpr']:.1%} of {t['n_ham']} held-out genuine SMS.
- Scam type, trained on synthetic and tested on reported: `{t['mt_name']}`, macro-F1 {t['mt_macro']:.3f}."""
    else:
        headline = ""
    data_note = """
| source | what it is | real? |
|---|---|---|
| `public` | UCI SMS Spam Collection (Almeida & Hidalgo 2011, CC BY 4.0), UK/Singapore SMS from ~2011 | **real**, not Indian |
| `reported` | written by us from scam messages described in cited Indian news, bank, police and government advisories (`public/reported_patterns.csv` has one source URL per row) | real patterns, our wording |
| `synthetic` | generated by `python -m text.synthetic` from seeded templates (Indian scams and genuine messages) | **synthetic** |
| `self` | `sample_messages.csv`, 36 test fixtures | synthetic, not used here |
"""
    sections = [
        ("Summary", f"""{banner}{HONEST_FRAMING} The scam-message corpus mixes one real public dataset with
messages written by us; no message was collected from a person. The table below says which is which.

Data: `{df.attrs['path']}`, {len(df):,} messages ({int(y_bin.sum()):,} scam, {int((1 - y_bin).sum()):,} genuine).
Generated by `python -m text.build_corpus` then `python -m text.classifier`.
{headline}
- An ordinary random 80/20 split gives macro-F1 {binary['models'][b_name]['test_macro_f1']:.3f} (binary,
  `{b_name}`) and {multi['models'][m_name]['test_macro_f1']:.3f} (type, `{m_name}`). These are **optimistic**:
  synthetic messages from the same template land on both sides of the split."""),
        ("Dataset", [data_note, by_source, counts, "UCI spam is labelled `other_scam` (it is generic spam, not one of "
                     "the Indian scam types). Classes with fewer than 5 messages merged into `other_scam`: "
                     + (", ".join(merged) if merged else "none") + "."]),
        ("Method", f"""
- (a) `nb_word`: MultinomialNB on word 1-2-gram counts; (b) `nb_char`: MultinomialNB on `char_wb`
  2-5-gram counts; (c) `nb_word_char`: both, via `FeatureUnion`; (d) `lr_tfidf`: LogisticRegression
  (`class_weight="balanced"`) on TF-IDF word + char features.
- Lowercased; the placeholder tokens are kept (`token_pattern = {TOKEN_PATTERN!r}`).
- Every split chooses NB `alpha` in {NB_ALPHAS} and LR `C` in {LR_CS} by stratified k-fold CV (macro-F1) on its
  training part only, then evaluates once.
- **Transfer test:** train on `synthetic` + 80% of `public`; test on all of `reported` + the other 20% of
  `public`. Because the reported messages were written independently of the templates, from real cases,
  this is the fairest number we can give without collected messages.
- **Random split:** stratified 80/20 by `scam_type` (seed {SEED}), for comparison."""),
    ]
    if transfer:
        sections += [
            ("Results: transfer test (headline)",
             ["Binary, all four models:", transfer["tab"],
              "Recall on the reported Indian scam messages, by type (best model):", transfer["rep_recall"],
              f"False alarms on held-out genuine UCI SMS: {transfer['fpr']:.1%}. Recall on held-out UCI spam: "
              f"{transfer['uci_spam_recall']:.1%}.",
              "Scam type (trained on synthetic, tested on reported):", transfer["mt_tab"], transfer["mt_per"],
              transfer["f_cm"]]),
        ]
    sections += [
        ("Results: random split (optimistic)", ["Binary:", bin_tab, "Scam type:", multi_tab]),
        ("Slice by language", ["Random split, binary, best model:", lang_tab]),
        ("Top scam-indicating features",
         ["`log P(t | scam) - log P(t | genuine)` from MultinomialNB (alpha = 1) fitted on every message. Words:",
          top_w, "Character n-grams:", top_c]),
        ("Naive Bayes theory", [THEORY, "Worked example (`nb_word`, alpha = 1, a reported message):",
                                worked_example(word_nb, example)]),
        ("Limitations", """
- **No collected messages.** The Indian part is written by us: the reported set from cited real cases,
  the synthetic set from templates. Real messages are messier (typos, mixed scripts, odd formatting).
- **Same author.** The reported and synthetic sets share a writer, so the transfer test is still
  easier than truly unseen messages would be.
- **UCI is old and not Indian.** It supplies real genuine SMS and generic spam from ~2011, not the
  scam types this project targets; its spam is labelled `other_scam`.
- **Small reported set:** 56 messages, so per-type recall moves a lot with one message.
- Placeholders (`<URL>`, `<PHONE>`) hide signal such as the link's domain.
- **Licence:** UCI SMS Spam Collection, CC BY 4.0, credit Almeida, T. & Hidalgo, J. (2011),
  UCI Machine Learning Repository, https://doi.org/10.24432/C5CC84."""),
        ("Parameters used", params_table(["SCAM_TYPES", "MIN_OWN_MESSAGES", "TEXT_MODEL_PATH"])),
    ]
    write_markdown_report(out, "Scam-message classifier: Naive Bayes", sections)
    print(f"wrote {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Person B scam-message classifier")
    ap.add_argument("--messages", default=None)
    ap.add_argument("--out", default=str(REPORT))
    a = ap.parse_args(argv)
    run_text_study(a.messages, a.out)


if __name__ == "__main__":
    main()
