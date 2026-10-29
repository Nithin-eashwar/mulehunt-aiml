# Chapter 4. Agent design

> Real transaction data is legally unavailable; we built a simulator calibrated to NPCI/RBI statistics and validated against the IBM AML benchmark.

Chapters 1-3 (Person A) built the data and the models that score each account. This chapter describes the *agent* that uses them: what it is trying to achieve, what it can see and do, and what kind of environment it works in. Chapters 5-8 then describe each of its parts.

## 4.1 PEAS

| | MuleHunt tracing and detection agent |
|---|---|
| **Performance** | money traced and recovered; ring members found (mules, cash-out); false alarms on genuine accounts |
| **Environment** | the UPI transaction network: accounts and timestamped payments between them |
| **Actuators** | flag, freeze or investigate an account (in our system: show it to a human analyst) |
| **Sensors** | the transaction stream, plus the classifier's risk score for each account |

## 4.2 Environment properties

| Property | MuleHunt | Why |
|---|---|---|
| Observability | partially observable | the bank sees payments, not intent; a sleeper mule looks normal until it forwards stolen money |
| Determinism | stochastic | whether a mule forwards money, when, and how much, is not under the agent's control |
| Episodes | sequential | freezing an account now changes which trails exist later |
| Dynamics | dynamic | new payments arrive while the agent deliberates; mules forward money within hours |
| State / time | continuous time, discrete accounts | we discretise time only through the hop window |
| Agents | multi-agent, adversarial | fraudsters adapt to the detector's rules (chapter 6) |

## 4.3 Agent types

- **Tracing is goal-based.** Starting from one flagged victim payment, the agent searches for a sequence of payments that reaches a known cash-out account (the goal). Any trail that reaches the goal is a success; among them, A\* prefers the most suspicious one (chapter 5).
- **The evasion game is utility-based.** The detector chooses a rule that minimises the money the fraudster can move *minus* the cost of false alarms on genuine customers. That trade-off is a utility, not a yes/no goal (chapter 6).
- **The classifiers are the agent's perception.** Person A's random forest turns raw history into a risk score per account. Search uses it as an edge cost (chapter 5), local search tunes how it is thresholded (chapter 6), and the dashboard shows it with its reasons (chapter 8).

## 4.4 How the parts form one system

```
transactions ──► features (A) ──► random forest (A) ──► risk score per account
     │                                                       │
     └──► time-respecting graph (B) ◄──────── edge cost ─────┘
              │
              ├── BFS / DFS / DLS / IDDFS / UCS / A*  ──► trail to cash-out, ring members found
              ├── minimax game  ──► which detection rule is worst-case safe
              └── local search  ──► which features to use, which threshold to flag at
scam messages ──► Naive Bayes ──► scam probability and type (chapter 7)
all of it     ──► Streamlit dashboard for a human analyst (chapter 8)
```

The design principle throughout is that **flags prompt human review**. Nothing in MuleHunt freezes an account automatically: the classifier misses a third of the mules, search touches some genuine accounts, and the game's model of the fraudster is simplified.
