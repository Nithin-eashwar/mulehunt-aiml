# MuleHunt demo script (3-5 minutes)

Before recording: `python -m app.precompute`, then `streamlit run app/streamlit_app.py`, and open each page
once so the graph is cached (the first load takes about 20 s).

| Time | Page | What to click | What to say | The one number |
|---|---|---|---|---|
| 0:00-0:30 | (title) | none | "Real transaction data is legally unavailable, so Nithin built a simulator calibrated to NPCI/RBI statistics and validated against the IBM AML benchmark. It has 50,377 accounts and 75 fraud rings. Our models flag mule accounts; my part follows the money." | 75 rings |
| 0:30-1:15 | Transaction check | Default demo: the `digital_arrest` victim payment | "Models score accounts, so a payment gets the higher of its two accounts' scores. The receiver here is a real tier-1 mule, but it scores 0.451, just under the 0.484 threshold. The classifier would let it through." Point at the explain() bars. | 0.451 < 0.484 |
| 1:15-2:30 | Mule trail explorer | Same payment, BFS, 48 h; then toggle **Reveal ground truth**; then switch the window to 12 h | "Search goes forward only along hops that happen after the money arrived and within 48 h. BFS reaches the cash-out in 3 hops after expanding 6 states. Revealed: the tier-1 and tier-3 mules on this trail were both missed by the classifier. At 12 h the trail breaks: one mule held the money for more than 12 hours before forwarding it." | trails recover all 455 mules; the classifier flags 309 |
| 2:30-3:30 | Evasion game | Policy *fixed T=50,000 velocity off*, then *minimax* | "A fixed ₹50k rule: the fraudster splits into parts under ₹50k and nothing is caught. Minimax picks ₹25k plus a velocity rule, and the fraudster loses ₹87,500. Alpha-beta visits 691 nodes instead of 18,631 for the same answer." | 96% fewer nodes |
| 3:30-4:15 | Scam message checker | Click *fake kyc*, then type a genuine message ("dinner ka paisa bhej diya") | "Victims reach the mules through messages like this. Naive Bayes on word and character n-grams handles Hinglish spelling. It was trained on real UCI SMS plus Indian messages we wrote, and on cited Indian scams it never saw it catches 54 of 56." | 54 / 56 cited scams caught |
| 4:15-4:45 | (any) | Footer | "Every flag is a prompt for a human analyst, not an automatic freeze: the classifier misses a third of mules and search can touch genuine accounts." | none |
