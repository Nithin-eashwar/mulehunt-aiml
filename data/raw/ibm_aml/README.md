# IBM AML benchmark: laundering patterns (HI-Small)

`HI-Small_Patterns.txt` comes from IBM's "Transactions for Anti Money Laundering (AML)" dataset
(Altman et al.), downloaded from Kaggle on 2026-10-27:
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

- **Licence:** Community Data License Agreement – Sharing, version 1.0 (CDLA-Sharing-1.0),
  https://cdla.dev/sharing-1-0/. The file is unmodified.
- **Contents:** 370 laundering attempts in 8 pattern types (BIPARTITE 49, CYCLE 54, FAN-IN 40,
  FAN-OUT 48, GATHER-SCATTER 51, RANDOM 41, SCATTER-GATHER 44, STACK 43).
- **How we use it:** structure only. `simulator/validate.py` checks that each IBM pattern type we
  implement appears in our fraud graph. We never use its amounts or transactions as training data
  (handoff §5.1).
