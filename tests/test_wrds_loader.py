import pandas as pd
import pytest

from riskparity.wrds_loader import CRSPRequest, build_crsp_query, total_return_index


def test_query_is_bounded_and_allowlisted():
    request = CRSPRequest.create(["SPY", "TLT"], "2020-01-01", "2020-12-31")
    query = build_crsp_query(request)
    assert "crsp.dsf" in query
    assert "crsp.dsenames" in query
    assert "'SPY', 'TLT'" in query
    assert "2020-01-01" in query and "2020-12-31" in query


def test_invalid_ticker_is_rejected():
    with pytest.raises(ValueError):
        CRSPRequest.create(["SPY'); DROP TABLE x;--"], "2020-01-01", "2020-12-31")


def test_total_return_index_combines_delisting_return():
    rows = pd.DataFrame({
        "ticker": ["SPY", "SPY"],
        "date": ["2020-01-02", "2020-01-03"],
        "ret": [0.10, 0.00],
        "dlret": [None, -0.50],
    })
    index = total_return_index(rows, ["SPY"])
    assert index.iloc[0, 0] == pytest.approx(110.0)
    assert index.iloc[1, 0] == pytest.approx(55.0)
