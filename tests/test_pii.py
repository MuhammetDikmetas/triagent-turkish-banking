import random

import pytest

from triagent.pii import is_valid_iban_tr, is_valid_tckn, luhn_ok, mask
from triagent.synthetic import decoy_number, fake_card, fake_iban, fake_phone, fake_tckn
from triagent.text import extract_amounts, fold

RNG = random.Random(7)


def test_generated_identifiers_pass_their_own_validators():
    for _ in range(200):
        assert is_valid_tckn(fake_tckn(RNG))
        assert is_valid_iban_tr(fake_iban(RNG))
        assert luhn_ok(fake_card(RNG))


def test_known_invalid_values_are_rejected():
    assert not is_valid_tckn("12345678901")
    assert not is_valid_tckn("02345678901")  # cannot start with 0
    assert not luhn_ok("4111 1111 1111 1112")
    assert luhn_ok("4111 1111 1111 1111")


def test_masks_every_type_in_one_message():
    tckn, iban, card, phone = fake_tckn(RNG), fake_iban(RNG), fake_card(RNG), fake_phone(RNG)
    text = (
        f"TC {tckn}, IBAN {iban}, kart {card}, tel {phone}, mail ali.veli@gmail.com. "
        "Acil dönüş yapın."
    )
    result = mask(text)
    for raw in (tckn, iban, card, phone, "ali.veli@gmail.com"):
        assert raw not in result.text
    assert result.counts == {"TCKN": 1, "IBAN": 1, "CARD": 1, "PHONE": 1, "EMAIL": 1}
    assert result.text.endswith("Acil dönüş yapın.")


def test_order_numbers_are_not_masked():
    decoy = decoy_number(RNG)
    text = f"Sipariş numaram {decoy}, referans kodu 4111111111111112."
    assert mask(text).entities == []


@pytest.mark.parametrize(
    "phone", ["0532 123 45 67", "+90 532 123 4567", "05321234567", "532-123-45-67"]
)
def test_phone_formats(phone):
    assert mask(f"Beni {phone} numarasından arayın").counts == {"PHONE": 1}


def test_turkish_folding():
    assert fold("KARTIM ÇALINDI") == "kartim calindi"
    assert fold("İPTAL edin") == "iptal edin"


def test_amount_extraction_turkish_format():
    assert extract_amounts("1.250,50 TL ve 300 lira ve 2000₺ kesildi") == [1250.5, 300.0, 2000.0]