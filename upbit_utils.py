# upbit_utils.py
import requests
import pandas as pd

# ----------------------------------
# 업비트 전체 티커 가져오기
# ----------------------------------
def get_upbit_tickers(market: str = "KRW"):
    """
    market: "KRW", "BTC" 등
    return: tickers(list), names(dict: {ticker: korean_name})
    """
    url = "https://api.upbit.com/v1/market/all"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    tickers = []
    names = {}

    for item in data:
        if item["market"].startswith(market):
            tickers.append(item["market"])
            names[item["market"]] = item["korean_name"]

    return tickers, names


# ----------------------------------
# OHLCV 데이터 가져오기 (일봉)
# ----------------------------------
def upbit_ohlcv(ticker: str, count: int = 200) -> pd.DataFrame:
    """
    ticker 예: "KRW-BTC"
    count: 가져올 일봉 개수
    """
    url = "https://api.upbit.com/v1/candles/days"
    params = {"market": ticker, "count": count}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["candle_date_time_kst"])
    df = df.set_index("date").sort_index()

    df = df.rename(columns={
        "opening_price": "Open",
        "high_price": "High",
        "low_price": "Low",
        "trade_price": "Close",
        "candle_acc_trade_volume": "Volume",
    })

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return df

import requests
import pandas as pd

def upbit_ohlcv(ticker: str, tf: str, count: int = 200):

    base = "https://api.upbit.com/v1/candles"

    # ---- 시간 프레임 라우팅 ----
    if tf == "일봉":
        url = f"{base}/days"
    elif tf == "주봉":
        url = f"{base}/weeks"
    elif "분봉" in tf:
        minute = tf.replace("분봉", "")  # "5분봉" → "5"
        url = f"{base}/minutes/{minute}"
    else:
        raise ValueError("지원하지 않는 시간 프레임입니다.")

    params = {"market": ticker, "count": count}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    # ---- DataFrame 정리 ----
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["candle_date_time_kst"])
    df = df.set_index("date").sort_index()

    df = df.rename(columns={
        "opening_price": "Open",
        "high_price": "High",
        "low_price": "Low",
        "trade_price": "Close",
        "candle_acc_trade_volume": "Volume",
    })

    return df[["Open", "High", "Low", "Close", "Volume"]]
