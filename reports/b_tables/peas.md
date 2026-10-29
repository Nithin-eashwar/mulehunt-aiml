# PEAS and environment (draft for chapter 04)

| | MuleHunt tracing / detection agent |
|---|---|
| Performance | money traced and recovered, ring members found, false alarms on genuine accounts |
| Environment | the UPI transaction network (accounts, timestamped payments) |
| Actuators | flag, freeze or investigate an account |
| Sensors | the transaction stream, plus model risk scores |

Environment type: partially observable (the bank sees payments, not intent), stochastic, sequential,
dynamic and multi-agent (fraudster vs detector).
Agent type: goal-based for tracing (reach a known cash-out), utility-based for the evasion game.
