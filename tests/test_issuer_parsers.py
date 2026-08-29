import pytest

from screener.providers.issuer_holdings import _parse_ishares_csv


def test_parse_ishares_csv_happy_path():
    content = (
        "iShares Fund,as of date\n"
        "some,preamble,rows\n"
        "Ticker,Name,Weight (%),Sector,Asset Class\n"
        "AAPL,Apple Inc,7.50,Technology,Equity\n"
    ).encode()
    df = _parse_ishares_csv(content)
    assert list(df["ticker"]) == ["AAPL"]
    assert df["weight_pct"].iloc[0] == 7.50


def test_parse_ishares_csv_raises_clear_error_when_format_changed():
    content = b"Something Else,Not A Header\nrow,1\n"
    with pytest.raises(ValueError, match="registry needs a refresh"):
        _parse_ishares_csv(content)
