import pytest

from text.anonymise import anonymise


@pytest.mark.parametrize("raw, token", [
    ("Call +91 98765 43210 now", "<PHONE>"),
    ("call 09876543210", "<PHONE>"),
    ("whatsapp 9876543210.", "<PHONE>"),
    ("ph 987-654-3210", "<PHONE>"),
    ("visit https://sbi-kyc.example.com/update?id=1", "<URL>"),
    ("click bit.ly/3xYz now", "<URL>"),
    ("open www.free-loan.in today", "<URL>"),
    ("go to rbi-refund.xyz", "<URL>"),
    ("pay to fraudster.99@ybl", "<UPI>"),
    ("mail help.desk@gmail.com", "<EMAIL>"),
    ("Your OTP is 482913", "<OTP>"),
    ("482913 is your OTP", "<OTP>"),
    ("PIN: 4821", "<OTP>"),
    ("a/c 123456789012 blocked", "<ACCOUNT>"),
    ("card 4111 1111 1111 1111", "<ACCOUNT>"),
])
def test_each_pattern(raw, token):
    clean, _ = anonymise(raw)
    assert token in clean
    assert not any(ch.isdigit() for ch in clean.replace(token, ""))


@pytest.mark.parametrize("raw", ["Pay Rs 5,000 fee", "₹1,50,000 approved", "INR 2500 cashback", "earn 3000 rupees",
                                 "Rs.9876543210 refund"])
def test_amounts_kept(raw):
    clean, _ = anonymise(raw)
    assert clean == raw


def test_idempotent():
    raw = "Dear customer, a/c 123456789012 KYC pending. Call 9876543210 or visit bit.ly/x. OTP 1234. Rs 10,000 fine"
    once, _ = anonymise(raw)
    twice, _ = anonymise(once)
    assert once == twice


def test_needs_review():
    assert anonymise("Dear Ramesh, your parcel is held")[1]
    assert anonymise("Mr. Sharma please pay")[1]
    assert anonymise("ref no 4521 pending")[1]  # digits left behind
    assert not anonymise("Dear Customer, your KYC expired. Pay Rs 500")[1]
    assert not anonymise("Call 9876543210")[1]
