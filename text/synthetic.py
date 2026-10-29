"""Seeded generator for synthetic Indian scam and genuine messages (source = "synthetic").

    python -m text.synthetic [--n-per-type 60] [--seed 0]

Templates are written by us from the scam patterns in reported_patterns.csv and from
ordinary bank / delivery / personal messages. Every row is labelled synthetic, so the
report can keep it apart from real data. Writes
data/raw/scam_messages/public/synthetic_messages.csv.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

OUT = Path("data/raw/scam_messages/public/synthetic_messages.csv")

SLOTS = {
    "bank": ["SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "PNB", "Kotak", "Canara Bank", "Bank of Baroda",
             "Union Bank", "IDFC First"],
    "telco": ["Jio", "Airtel", "Vi", "BSNL"],
    "utility": ["MSEDCL", "BESCOM", "TPDDL", "electricity board", "power department", "Adani Electricity"],
    "agency": ["CBI", "Mumbai Police", "Narcotics Control Bureau", "Customs", "ED", "Cyber Crime Cell", "TRAI",
               "RBI"],
    "courier": ["FedEx", "DTDC", "Blue Dart", "DHL"],
    "country": ["Taiwan", "Cambodia", "Thailand", "Iran", "Dubai"],
    "item": ["drugs", "MDMA", "fake passports", "illegal SIM cards", "foreign currency"],
    "brand": ["Amazon", "Flipkart", "Meesho", "Myntra", "a 5-star hotel chain", "YouTube"],
    "app": ["QuickCash", "RupeeFast", "LoanTurbo", "CashKaro Now", "EasyPaisa Loan", "MoneyNow"],
    "hours": ["2", "12", "24", "48"],
    "time": ["9:30 pm", "10 pm", "tonight", "today", "within 1 hour", "by midnight"],
    "amt_s": ["499", "999", "1,500", "1,999", "2,500", "3,000", "5,000"],
    "amt_m": ["8,000", "12,500", "25,000", "49,000", "75,000"],
    "amt_l": ["1,50,000", "2,00,000", "3,50,000", "5,00,000", "10,00,000"],
    "pct": ["20%", "30%", "50%", "200%", "300%"],
    "days": ["7", "15", "30"],
    "name": ["Rahul", "Priya", "Amit", "Neha", "Arjun", "Sneha", "Vikram", "Pooja"],
    "shop": ["Big Bazaar", "DMart", "Reliance Fresh", "Swiggy", "Zomato", "BookMyShow", "IRCTC", "BigBasket"],
    "rel": ["Mummy", "Papa", "Didi", "Bhaiya", "Beta"],
}

SCAM = {
    "fake_kyc": [
        ("english", "sms", "Dear customer, your {bank} account KYC has expired. Account will be blocked {time}. Update PAN at <URL>"),
        ("english", "sms", "{bank}: Your NetBanking access will be suspended in {hours} hrs. Complete KYC now <URL>"),
        ("english", "whatsapp", "{bank} alert: KYC pending. Download {bank}_KYC.apk and verify to keep your account active."),
        ("english", "sms", "Dear {telco} user, your SIM KYC is incomplete. SIM will be blocked in {hours} hours. Call <PHONE>"),
        ("english", "sms", "Dear consumer, power will be disconnected {time} as your bill is not updated. Contact {utility} officer <PHONE>"),
        ("english", "sms", "{utility}: your electricity KYC is pending. Disconnection {time}. Update via <URL>"),
        ("hinglish", "sms", "Priya grahak, aapka {bank} KYC expire ho gaya hai. {time} account block hoga. Turant <URL> par update karein."),
        ("hinglish", "whatsapp", "Aapka {telco} SIM {hours} ghante mein band ho jayega. KYC ke liye <PHONE> par call karein."),
        ("hinglish", "sms", "Bijli connection {time} kat jayega, bill update nahi hua. {utility} officer se baat karein <PHONE>"),
        ("hindi", "sms", "प्रिय ग्राहक, आपका {bank} खाता KYC अपडेट न होने के कारण {time} बंद हो जाएगा। तुरंत <URL> पर अपडेट करें।"),
        ("hindi", "sms", "आपका बिजली कनेक्शन {time} काट दिया जाएगा। तुरंत अधिकारी से संपर्क करें <PHONE>"),
    ],
    "digital_arrest": [
        ("english", "other", "This is {courier}. A parcel in your name to {country} containing {item} has been seized. Press 1 to speak to the officer."),
        ("english", "whatsapp", "{agency} notice: a case is registered against your Aadhaar. Join video call for statement. Do not tell anyone."),
        ("english", "other", "You are under digital arrest by {agency}. Stay on the call. Transfer your balance to the verification account for checking."),
        ("english", "whatsapp", "Arrest warrant issued against you by {agency}. Pay Rs {amt_l} security deposit to avoid arrest today."),
        ("english", "other", "Your mobile number will be disconnected in {hours} hours due to illegal activity reported by {agency}. Press 9."),
        ("english", "whatsapp", "Money laundering case linked to your bank account. {agency} officer will call you on video. Keep it confidential."),
        ("hinglish", "other", "Main {agency} se bol raha hoon. Aapke Aadhaar se {country} parcel gaya hai jisme {item} mila hai. Call mat kaatna."),
        ("hinglish", "whatsapp", "Aap digital arrest mein ho. Video call par bane raho aur Rs {amt_l} RBI safe account mein bhejo."),
        ("hindi", "other", "आपके नाम से {country} भेजे गए पार्सल में {item} मिला है। {agency} अधिकारी से बात करने के लिए 1 दबाएं।"),
        ("hindi", "whatsapp", "आपके खिलाफ गिरफ्तारी वारंट जारी हुआ है। वीडियो कॉल से जुड़ें और किसी को न बताएं।"),
    ],
    "job_task": [
        ("english", "whatsapp", "Part-time job from {brand}! Earn Rs {amt_s} to Rs {amt_m} per day by liking videos. No joining fee. Reply YES."),
        ("english", "whatsapp", "Hi, HR here. Rate hotels and earn Rs 150 per task. Daily payment. Join our Telegram <URL>"),
        ("english", "sms", "You are selected for a {brand} work from home role. Salary Rs {amt_m}/month. Pay registration fee Rs {amt_s} <URL>"),
        ("english", "whatsapp", "Task 4 complete! Recharge Rs {amt_m} to unlock VIP tasks and withdraw your Rs {amt_m} profit."),
        ("english", "whatsapp", "Your task wallet is frozen due to wrong operation. Pay Rs {amt_m} to recover your funds."),
        ("hinglish", "whatsapp", "Ghar baithe roz Rs {amt_s} kamao, sirf YouTube videos like karne hain. Interested ho to YES likho."),
        ("hinglish", "whatsapp", "Sir premium task ke liye Rs {amt_m} deposit karo, {pct} commission milega aaj hi."),
        ("hinglish", "sms", "{brand} mein part time job, roz Rs {amt_s} payment. Training fee Rs {amt_s} pehle bhejein <UPI>"),
        ("hindi", "whatsapp", "घर बैठे रोज़ Rs {amt_s} कमाएं। सिर्फ वीडियो लाइक करने हैं। अभी जुड़ें <URL>"),
    ],
    "loan_app": [
        ("english", "sms", "Instant loan up to Rs {amt_l} approved without CIBIL. Download {app} <URL> and get money in 5 minutes."),
        ("english", "sms", "Pre-approved loan Rs {amt_l} at 0% interest. Pay processing fee Rs {amt_s} to disburse today."),
        ("english", "whatsapp", "Your {app} loan is overdue. Pay Rs {amt_m} now or your photos will be sent to all your contacts."),
        ("english", "whatsapp", "We have your contacts and gallery. Repay Rs {amt_m} today or we will message your family and office."),
        ("english", "sms", "{app}: Loan of Rs {amt_m} credited. Repay Rs {amt_m} within {days} days. Late fee Rs 1,000 per day."),
        ("hinglish", "whatsapp", "{app} ka paisa abhi bhejo warna teri photos sab contacts ko bhej denge."),
        ("hinglish", "sms", "Sirf Aadhaar se Rs {amt_l} ka loan. Processing charge Rs {amt_s} <UPI> par bhejein."),
        ("hindi", "whatsapp", "लोन की राशि तुरंत चुकाएं वरना आपकी फोटो सभी कॉन्टैक्ट्स को भेज दी जाएगी।"),
    ],
    "fake_investment": [
        ("english", "whatsapp", "Join our VIP stock group. Members earned {pct} returns in {days} days. Invest via <URL>"),
        ("english", "whatsapp", "Exclusive IPO allotment for group members. Deposit Rs {amt_l} in the institutional account to get shares."),
        ("english", "whatsapp", "Crypto trading bot doubles your money in {days} days. Minimum investment Rs {amt_m}. Guaranteed returns."),
        ("english", "whatsapp", "Today our members made Rs {amt_l} profit! Transfer to fund manager <UPI> for tomorrow's call."),
        ("english", "sms", "Gold scheme: invest Rs {amt_m}, get {pct} monthly return. Limited slots. Call <PHONE>"),
        ("hinglish", "whatsapp", "Group ke members ne aaj {pct} profit kamaya. Aap bhi Rs {amt_m} invest karo, double pakka."),
        ("hinglish", "whatsapp", "Trading mentor ki free class join karo, roz ka buy/sell call milega. Link <URL>"),
        ("hindi", "whatsapp", "{days} दिन में पैसा दोगुना। अभी Rs {amt_m} निवेश करें। लिंक <URL>"),
    ],
    "other_scam": [
        ("english", "sms", "Your vehicle was caught overspeeding. Pay challan Rs {amt_s} {time} <URL> to avoid licence suspension."),
        ("english", "sms", "India Post: your parcel is held due to incomplete address. Update within {hours} hours <URL>"),
        ("english", "sms", "You are approved for an income tax refund of Rs {amt_m}. Verify your bank account <URL>"),
        ("english", "sms", "Your {bank} credit card reward points worth Rs {amt_s} expire {time}. Redeem <URL>"),
        ("english", "sms", "Congratulations! You won Rs {amt_l} in KBC lucky draw. Pay tax Rs {amt_s} to claim. Call <PHONE>"),
        ("english", "sms", "Your FASTag is blocked. Recharge via <URL> to avoid penalty."),
        ("hinglish", "sms", "Aapka challan pending hai, Rs {amt_s} {time} bharein warna court notice aayega <URL>"),
        ("hindi", "sms", "आपका पार्सल पता अधूरा होने के कारण रुका है। {hours} घंटे में पता अपडेट करें <URL>"),
    ],
}

GENUINE = [
    ("english", "sms", "Rs {amt_s} debited from your {bank} a/c <ACCOUNT> for UPI txn to {shop}. Not you? Call the number on your card."),
    ("english", "sms", "Rs {amt_m} credited to your {bank} a/c <ACCOUNT>. Available balance updated."),
    ("english", "sms", "<OTP> is your OTP for {shop} payment of Rs {amt_s}. Do not share it with anyone. {bank} never asks for OTP."),
    ("english", "sms", "Your {bank} KYC update is complete. Thank you for banking with us."),
    ("english", "sms", "Your {telco} bill of Rs {amt_s} is generated. Due date in {days} days. Pay on the official app."),
    ("english", "sms", "Electricity bill of Rs {amt_s} generated for your connection. Due in {days} days. Pay via official {utility} portal."),
    ("english", "sms", "Your order from {shop} has been delivered. Rate your experience in the app."),
    ("english", "sms", "Your {shop} order is out for delivery. Delivery partner will call you."),
    ("english", "sms", "PNR status: your train ticket booked via IRCTC is confirmed. Happy journey."),
    ("english", "sms", "Salary of Rs {amt_m} credited to your account. - {bank}"),
    ("english", "sms", "Your EMI of Rs {amt_m} is due on the 5th. Please keep sufficient balance in your {bank} account."),
    ("english", "whatsapp", "Hi {name}, meeting moved to 4 pm tomorrow. Please bring the report."),
    ("english", "whatsapp", "Happy birthday {name}! Have a great year ahead."),
    ("english", "whatsapp", "Reached office. Will call you at lunch."),
    ("english", "whatsapp", "Can you send the notes from today's lecture?"),
    ("english", "sms", "Your appointment with Dr. Sharma is confirmed for tomorrow 11 am."),
    ("english", "sms", "Dear customer, your {telco} recharge of Rs {amt_s} is successful. Validity {days} days."),
    ("hinglish", "whatsapp", "{rel}, ghar aate waqt doodh le aana."),
    ("hinglish", "whatsapp", "Bhai dinner ke Rs {amt_s} bhej diye, check kar lena."),
    ("hinglish", "whatsapp", "Kal college kitne baje aana hai? Assignment submit karna hai."),
    ("hinglish", "whatsapp", "{name}, train late hai, 8 baje tak pahunchunga."),
    ("hinglish", "whatsapp", "{rel} ne bola Sunday ko sab ghar aa jao, khana yahin hai."),
    ("hinglish", "sms", "Aapka order deliver ho gaya hai. {shop} ko rate karein."),
    ("hindi", "sms", "आपके {bank} खाते से Rs {amt_s} UPI द्वारा {shop} को भुगतान किया गया।"),
    ("hindi", "whatsapp", "शाम को मंदिर चलेंगे, तैयार रहना।"),
    ("hindi", "sms", "आपका बिजली बिल Rs {amt_s} जारी किया गया है। अंतिम तिथि {days} दिन में।"),
]


def _fill(template: str, rng) -> str:
    out = template
    for key, values in SLOTS.items():
        while "{" + key + "}" in out:
            out = out.replace("{" + key + "}", values[int(rng.integers(len(values)))], 1)
    return out


def generate(n_per_type=60, n_genuine=200, seed=0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows, seen = [], set()

    def add(templates, scam_type, n):
        tries = 0
        made = 0
        while made < n and tries < n * 50:
            tries += 1
            lang, channel, t = templates[int(rng.integers(len(templates)))]
            text = _fill(t, rng)
            if text in seen:
                continue
            seen.add(text)
            rows.append({"text": text, "scam_type": scam_type, "language": lang, "channel": channel})
            made += 1

    for scam_type, templates in SCAM.items():
        add(templates, scam_type, n_per_type)
    add(GENUINE, "genuine", n_genuine)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate synthetic Indian scam / genuine messages")
    ap.add_argument("--n-per-type", type=int, default=60)
    ap.add_argument("--n-genuine", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    rows = generate(a.n_per_type, a.n_genuine, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["ref_id", "text", "scam_type", "language", "channel"], lineterminator="\n")
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow({"ref_id": f"GEN{i:04d}", **r})
    print(f"wrote {a.out}: {len(rows)} synthetic messages")


if __name__ == "__main__":
    main()
