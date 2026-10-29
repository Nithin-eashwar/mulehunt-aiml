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


def run_text_study(messages=None, out=REPORT, fig_dir=FIGURES, tab_dir=TABLES, model_path=TEXT_MODEL_PATH):
    from mulehunt.report import write_markdown_report
    fig_dir, tab_dir = Path(fig_dir), Path(tab_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    df = load_messages(messages)
    synthetic = df.attrs["synthetic"]
    own = df[df.source != "public"]
    texts = df.text.tolist()
    y_bin = (df.label == "scam").astype(int).to_numpy()
    y_multi, merged = multiclass_labels(df)

    counts = pd.concat([df.scam_type.value_counts().rename("messages").rename_axis("value").reset_index()
                        .assign(by="scam_type"),
                        df.language.value_counts().rename("messages").rename_axis("value").reset_index()
                        .assign(by="language"),
                        df.channel.value_counts().rename("messages").rename_axis("value").reset_index()
                        .assign(by="channel")])[["by", "value", "messages"]]
    counts.to_csv(tab_dir / "text_counts.csv", index=False)

    binary = fit_task(texts, y_bin, df.scam_type)
    multi = fit_task(texts, y_multi.to_numpy(), y_multi)
    bin_tab, multi_tab = _results_table(binary), _results_table(multi)
    bin_tab.to_csv(tab_dir / "text_binary.csv", index=False)
    multi_tab.to_csv(tab_dir / "text_multiclass.csv", index=False)
    b_name, m_name = _best(binary), _best(multi)

    # per-class metrics and confusion matrix of the best multiclass model
    mres = multi["models"][m_name]
    labels = sorted(set(y_multi))
    per_class = pd.DataFrame(classification_report(multi["yte"], mres["pred"], labels=labels, output_dict=True,
                                                   zero_division=0)).T.round(3).reset_index(names="class")
    per_class.to_csv(tab_dir / "text_per_class.csv", index=False)
    f_cm = _confusion_figure(multi["yte"], mres["pred"], labels, fig_dir / "text_confusion.png")

    # language slice (binary, hold-out)
    bres = binary["models"][b_name]
    slice_rows = []
    hold = pd.DataFrame({"text": binary["Xte"], "y": binary["yte"], "pred": bres["pred"]})
    hold["language"] = hold.text.map(dict(zip(df.text, df.language)))
    for lg, g in hold.groupby("language"):
        slice_rows.append({"language": lg, "messages": len(g), "accuracy": round((g.y == g.pred).mean(), 3),
                           "macro_f1": round(f1_score(g.y, g.pred, average="macro", zero_division=0), 3)})
    lang_tab = pd.DataFrame(slice_rows)
    lang_tab.to_csv(tab_dir / "text_by_language.csv", index=False)

    # top scam-indicating features (NB fitted on all messages)
    word_nb = build_models()["nb_word"][0].set_params(model__alpha=1.0).fit(texts, y_bin)
    char_nb = build_models()["nb_char"][0].set_params(model__alpha=1.0).fit(texts, y_bin)
    top_w, top_c = top_features(word_nb), top_features(char_nb)
    top_w.to_csv(tab_dir / "text_top_words.csv", index=False)
    top_c.to_csv(tab_dir / "text_top_chars.csv", index=False)
    example = df[df.label == "scam"].text.iloc[0]

    # save the best pipelines, refit on every message
    bin_pipe = binary["models"][b_name]["search"].best_estimator_.fit(texts, y_bin)
    multi_pipe = multi["models"][m_name]["search"].best_estimator_.fit(texts, y_multi)
    save_artifact(bin_pipe, multi_pipe, labels, len(df), model_path)

    banner = ("**PRELIMINARY: synthetic fixtures.** The real form data is not collected yet; these numbers come "
              "from ~36 messages we wrote ourselves (`sample_messages.csv`) and only show that the pipeline "
              "works. Rerun `python -m text.classifier` once `messages.csv` exists.\n\n") if synthetic else ""
    supplement = ("Public supplement (B-D17): not used" + (f"; only {len(own)} own messages, so it applies once "
                  "real data arrives and stays below 300." if len(own) < 300 else "."))
    sections = [
        ("Summary", f"""{banner}{HONEST_FRAMING} The scam-message dataset is the one fully real dataset in the
project (once collected).

Data: `{df.attrs['path']}`, {len(df)} messages ({int(y_bin.sum())} scam, {int((1 - y_bin).sum())} genuine).
Generated by `python -m text.classifier`.

- Binary (scam vs genuine): best model `{b_name}`, CV macro-F1 {binary['models'][b_name]['cv_macro_f1']:.3f},
  hold-out macro-F1 {binary['models'][b_name]['test_macro_f1']:.3f}.
- Multiclass (scam type): best model `{m_name}`, CV macro-F1 {multi['models'][m_name]['cv_macro_f1']:.3f},
  hold-out macro-F1 {multi['models'][m_name]['test_macro_f1']:.3f}.
- {supplement}"""),
        ("Dataset", [counts, "Classes with fewer than 5 messages merged into `other_scam` for the multiclass task: "
                     + (", ".join(merged) if merged else "none") + "."]),
        ("Method", f"""
- (a) `nb_word`: MultinomialNB on word 1-2-gram counts; (b) `nb_char`: MultinomialNB on `char_wb`
  2-5-gram counts; (c) `nb_word_char`: both, via `FeatureUnion`; (d) `lr_tfidf`: LogisticRegression
  (`class_weight="balanced"`) on TF-IDF word + char features.
- Lowercased; the placeholder tokens are kept (`token_pattern = {TOKEN_PATTERN!r}`).
- Stratified 80/20 hold-out by `scam_type` (seed {SEED}); stratified k-fold CV (k = min(5, smallest class))
  on the 80% chooses NB `alpha` in {NB_ALPHAS} and LR `C` in {LR_CS} by macro-F1."""),
        ("Results: binary", bin_tab),
        ("Results: multiclass", [multi_tab, f"Per class (`{m_name}`, hold-out):", per_class, f_cm]),
        ("Slice by language", ["Binary hold-out, best model:", lang_tab]),
        ("Top scam-indicating features",
         ["`log P(t | scam) - log P(t | genuine)` from MultinomialNB (alpha = 1) fitted on every message. Words:",
          top_w, "Character n-grams:", top_c]),
        ("Naive Bayes theory", [THEORY, "Worked example (`nb_word`, alpha = 1):", worked_example(word_nb, example)]),
        ("Limitations", """
- Small n: a few hundred messages at best; hold-out numbers have wide uncertainty.
- Self-selection: contributors choose which messages to share, so blatant scams are likely
  over-represented and genuine messages under-represented.
- Labels are the contributors' own judgement (with "not sure" mapped to `other_scam`).
- The anonymiser plus human review removes personal data, but placeholders also hide some signal
  (e.g. which domain a link points to)."""),
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
