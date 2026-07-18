"""yfinance-based data adapter for the AI Hedge Fund.

Replaces the financialdatasets.ai API with yfinance as the data source.
No API key required — yfinance fetches data from Yahoo Finance for free.

Functions maintained with identical signatures for backward compatibility:
- get_prices, get_financial_metrics, search_line_items
- get_insider_trades, get_company_news, get_market_cap
- prices_to_df, get_price_data
"""
import datetime
import logging

import pandas as pd

logger = logging.getLogger(__name__)

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    Price,
    PriceResponse,
)

_cache = get_cache()

# ── yfinance field-to-row mappings ───────────────────────────────────────
# Map internal field names to yfinance financial statement row labels.

_IS_LABELS = {  # Income statement
    "revenue":                       "Total Revenue",
    "cost_of_revenue":               "Cost Of Revenue",
    "gross_profit":                  "Gross Profit",
    "operating_expense":             "Operating Expense",
    "research_and_development":      "Research And Development",
    "depreciation_and_amortization": "Depreciation And Amortization",
    "operating_income":              "Operating Income",
    "interest_expense":              "Interest Expense",
    "ebit":                          "EBIT",
    "ebitda":                        "EBITDA",
    "net_income":                    "Net Income",
    "earnings_per_share":            "Diluted EPS",
    "outstanding_shares":            "Diluted Average Shares",
}

_BS_LABELS = {  # Balance sheet
    "total_assets":           "Total Assets",
    "total_liabilities":      "Total Liabilities Net Minority Interest",
    "current_assets":         "Current Assets",
    "current_liabilities":    "Current Liabilities",
    "shareholders_equity":    "Stockholders Equity",
    "total_debt":             "Total Debt",
    "cash_and_equivalents":   "Cash And Cash Equivalents",
    "working_capital":        "Working Capital",
    "intangible_assets":      "Intangible Assets",
    "goodwill_and_intangible_assets": "Goodwill And Intangible Assets",
}

_CF_LABELS = {  # Cash flow
    "free_cash_flow":                    "Free Cash Flow",
    "capital_expenditure":               "Capital Expenditure",
    "dividends_and_other_cash_distributions": "Dividends Paid",
    "issuance_or_purchase_of_equity_shares":  "Repurchase Of Capital Stock",
}

_ALL_LABELS = {**_IS_LABELS, **_BS_LABELS, **_CF_LABELS}


# ── helpers ─────────────────────────────────────────────────────────────

def _lookup(statement, field):
    """Read *field* (our name) from *statement* (yfinance DataFrame)."""
    label = _ALL_LABELS.get(field)
    if label is None or statement is None or statement.empty:
        return None
    if label in statement.index:
        val = statement.loc[label]
        return float(val) if pd.notna(val) else None
    return None


def _safe_divide(a, b):
    """Return a / b or None if b is 0 or None."""
    if a is None or b is None or b == 0:
        return None
    return a / b


# ── derived field computations used by agents ──────────────────────────
# These fields are expected on LineItem objects but may not be explicitly
# requested.  We compute them whenever the prerequisite fields exist.

_DERIVED_FIELDS = [
    "book_value_per_share",
    "free_cash_flow",
    "gross_margin",
    "operating_margin",
    "return_on_invested_capital",
    "debt_to_equity",
]


def _compute_derived(data: dict) -> dict:
    """Compute derived fields and return them as a dict."""
    out = {}
    fcf = data.get("free_cash_flow")
    if fcf is not None:
        out["free_cash_flow"] = fcf
    # book_value_per_share
    eq = data.get("shareholders_equity")
    sh = data.get("outstanding_shares")
    bv = _safe_divide(eq, sh)
    if bv is not None:
        out["book_value_per_share"] = bv
    # gross_margin
    gp = data.get("gross_profit")
    rv = data.get("revenue")
    gm = _safe_divide(gp, rv)
    if gm is not None:
        out["gross_margin"] = gm
    # operating_margin
    oi = data.get("operating_income")
    om = _safe_divide(oi, rv)
    if om is not None:
        out["operating_margin"] = om
    # return_on_invested_capital
    ni = data.get("net_income")
    ta = data.get("total_assets")
    cl = data.get("current_liabilities")
    if ni is not None and ta is not None and cl is not None:
        invested_capital = ta - cl
        if invested_capital > 0:
            out["return_on_invested_capital"] = ni / invested_capital
    # debt_to_equity
    td = data.get("total_debt")
    eq2 = data.get("shareholders_equity")
    de = _safe_divide(td, eq2)
    if de is not None:
        out["debt_to_equity"] = de
    return out


# ── public API ──────────────────────────────────────────────────────────

def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    """Fetch price data from yfinance, with cache."""
    cache_key = f"{ticker}_{start_date}_{end_date}"
    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]

    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
    except Exception as e:
        logger.warning("yfinance get_prices(%s): %s", ticker, e)
        return []

    if df.empty:
        return []

    prices = [
        Price(
            open=float(row["Open"]),
            close=float(row["Close"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            volume=int(row["Volume"]),
            time=idx.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        for idx, row in df.iterrows()
    ]

    _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from yfinance .info data."""
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"
    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**m) for m in cached_data]

    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info or {}
    except Exception as e:
        logger.warning("yfinance get_financial_metrics(%s): %s", ticker, e)
        return []

    kw = {
        "ticker": ticker,
        "report_period": end_date,
        "period": period,
        "currency": info.get("financialCurrency", "USD"),
        "market_cap": info.get("marketCap"),
        "enterprise_value": info.get("enterpriseValue"),
        "price_to_earnings_ratio": info.get("trailingPE"),
        "price_to_book_ratio": info.get("priceToBook"),
        "price_to_sales_ratio": info.get("priceToSalesTrailing12Months"),
        "enterprise_value_to_ebitda_ratio": info.get("enterpriseToEbitda"),
        "enterprise_value_to_revenue_ratio": info.get("enterpriseToRevenue"),
        "free_cash_flow_yield": info.get("freeCashflowYield"),
        "peg_ratio": info.get("pegRatio"),
        "gross_margin": info.get("grossMargins"),
        "operating_margin": info.get("operatingMargins"),
        "net_margin": info.get("profitMargins"),
        "return_on_equity": info.get("returnOnEquity"),
        "return_on_assets": info.get("returnOnAssets"),
        "return_on_invested_capital": info.get("returnOnInvestedCapital"),
        "current_ratio": info.get("currentRatio"),
        "quick_ratio": info.get("quickRatio"),
        "cash_ratio": info.get("cashRatio"),
        "debt_to_equity": info.get("debtToEquity"),
        "debt_to_assets": info.get("debtToAssets"),
        "interest_coverage": info.get("interestCoverage"),
        "payout_ratio": info.get("payoutRatio"),
        "earnings_per_share": info.get("trailingEps"),
        "book_value_per_share": info.get("bookValue"),
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        "earnings_per_share_growth": info.get("earningsPerShareGrowth"),
        "asset_turnover": info.get("assetTurnover"),
        "inventory_turnover": info.get("inventoryTurnover"),
        "receivables_turnover": info.get("receivablesTurnover"),
        "free_cash_flow_per_share": info.get("freeCashflowPerShare"),
        "free_cash_flow_growth": None,
        "book_value_growth": None,
        "operating_income_growth": None,
        "ebitda_growth": None,
        "days_sales_outstanding": None,
        "operating_cycle": None,
        "working_capital_turnover": None,
        "operating_cash_flow_ratio": None,
    }
    
    metric = FinancialMetrics(**kw)
    result = [metric]
    _cache.set_financial_metrics(cache_key, [m.model_dump() for m in result])
    return result


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """Fetch financial line items from yfinance annual or quarterly statements.

    Returns one LineItem per annual period (or one for TTM).
    Derived fields commonly used by agents are computed automatically.
    """
    cache_key = f"{ticker}_{'_'.join(sorted(line_items))}_{end_date}_{period}_{limit}"
    if cached_data := _cache.get_line_items(cache_key):
        return [LineItem(**li) for li in cached_data]

    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
    except Exception as e:
        logger.warning("yfinance search_line_items(%s): %s", ticker, e)
        return []

    # Determine which statements cover each requested field
    is_fields = {f for f in line_items if f in _IS_LABELS}
    bs_fields = {f for f in line_items if f in _BS_LABELS}
    cf_fields = {f for f in line_items if f in _CF_LABELS}
    explicit_fields = is_fields | bs_fields | cf_fields

    if period == "ttm":
        results = _build_ttm_line_item(stock, explicit_fields, ticker, end_date)
    else:
        results = _build_annual_line_items(stock, explicit_fields, ticker, end_date, limit)

    if results:
        _cache.set_line_items(cache_key, [r.model_dump() for r in results])
    return results


# ── internal builders ───────────────────────────────────────────────────

def _build_annual_line_items(stock, explicit_fields, ticker, end_date, limit):
    """Build LineItem list from annual yfinance statements, one per year up to *limit*."""
    financials = stock.financials if hasattr(stock, "financials") and stock.financials is not None else pd.DataFrame()
    balance_sheet = stock.balance_sheet if hasattr(stock, "balance_sheet") and stock.balance_sheet is not None else pd.DataFrame()
    cashflow = stock.cashflow if hasattr(stock, "cashflow") and stock.cashflow is not None else pd.DataFrame()

    # Collect all column dates across statements
    all_dates = set()
    for stmt in (financials, balance_sheet, cashflow):
        if not stmt.empty:
            all_dates.update(stmt.columns)
    all_dates = sorted(all_dates, reverse=True)[:limit]

    results = []
    for period_date in all_dates:
        period_key = period_date.strftime("%Y-%m-%d") if hasattr(period_date, "strftime") else str(period_date)[:10]
        data = {"ticker": ticker, "report_period": period_key, "period": "annual", "currency": "USD"}

        # Read explicit + derived fields from statements
        all_needed = explicit_fields | set(_DERIVED_FIELDS) | {"outstanding_shares"}
        for field in all_needed:
            if field in _DERIVED_FIELDS or field in data:
                continue
            val = None
            for stmt, label_map in [(financials, _IS_LABELS), (balance_sheet, _BS_LABELS), (cashflow, _CF_LABELS)]:
                if stmt.empty:
                    continue
                label = label_map.get(field)
                if label is not None and label in stmt.index and period_date in stmt.columns:
                    raw = stmt.loc[label, period_date]
                    val = float(raw) if pd.notna(raw) else None
                    break
            data[field] = val

        # Compute derived fields
        data.update(_compute_derived(data))
        # Ensure all requested fields are present (even None) so agents can access them safely
        for field in all_needed | set(_DERIVED_FIELDS) | {"outstanding_shares"}:
            if field not in data:
                data[field] = None

        results.append(LineItem(**data))

    return results


def _build_ttm_line_item(stock, explicit_fields, ticker, end_date):
    """Build a single LineItem from trailing-twelve-months (sum of last 4 quarters)."""
    qf = stock.quarterly_financials if hasattr(stock, "quarterly_financials") and stock.quarterly_financials is not None else pd.DataFrame()
    qb = stock.quarterly_balance_sheet if hasattr(stock, "quarterly_balance_sheet") and stock.quarterly_balance_sheet is not None else pd.DataFrame()
    qc = stock.quarterly_cashflow if hasattr(stock, "quarterly_cashflow") and stock.quarterly_cashflow is not None else pd.DataFrame()

    data = {"ticker": ticker, "report_period": end_date, "period": "ttm", "currency": "USD"}
    all_needed = explicit_fields | set(_DERIVED_FIELDS) | {"outstanding_shares"}

    for field in all_needed:
        if field in _DERIVED_FIELDS:
            continue

        # IS/CF items: sum last 4 quarters
        for stmt, label_map in [(qf, _IS_LABELS), (qc, _CF_LABELS)]:
            if stmt.empty:
                continue
            label = label_map.get(field)
            if label is not None and label in stmt.index:
                series = stmt.loc[label]
                vals = [float(v) for v in series.head(4) if pd.notna(v)]
                if vals:
                    data[field] = sum(vals)
                break
        if field in data:
            continue

        # BS items: most recent quarter-end value
        if field in _BS_LABELS:
            if not qb.empty:
                label = _BS_LABELS.get(field)
                if label is not None and label in qb.index:
                    raw = qb.loc[label].iloc[0]
                    data[field] = float(raw) if pd.notna(raw) else None

    data.update(_compute_derived(data))
    # Ensure all requested fields are present (even if None) so agents can safely access them
    for field in all_needed | set(_DERIVED_FIELDS):
        if field not in data:
            data[field] = None
    return [LineItem(**data)] if any(k not in ("ticker", "report_period", "period", "currency") for k in data) else []


# ── stubs for data not available from yfinance ──────────────────────────

def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    """yfinance does not provide insider-trade data. Returns empty list."""
    return []


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    """yfinance does not provide structured news. Returns empty list."""
    return []


def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    """Fetch market cap from yfinance info."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        return info.get("marketCap")
    except Exception as e:
        logger.warning("yfinance get_market_cap(%s): %s", ticker, e)
        return None


# ── DataFrame helpers ──────────────────────────────────────────────────

def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices list to a DataFrame indexed by Date."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    """Convenience: get yfinance prices and return as DataFrame."""
    return prices_to_df(get_prices(ticker, start_date, end_date, api_key=api_key))
