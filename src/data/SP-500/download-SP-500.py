import yfinance as yf
import pandas as pd
import wikipedia
import os
from tqdm import tqdm

# Get the table of S&P 500 stocks from Wikipedia to access tickers
html = wikipedia.page("List of S&P 500 companies").html().encode("UTF-8")
table = pd.read_html(html)[1]
tickers = table["Symbol"].values


# Get Adjusted Close data for stock
def load_data(ticker, start="2000-01-01", end="2025-01-01", interval="1d"):
    try:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        if data.empty:
            raise ValueError(f"No data found for {ticker}")
        data = data[["Adj Close"]]
        data.dropna(inplace=True)
        return data
    except Exception as e:
        raise ValueError(f"Failed to load data for {ticker}: {e}")


# Get returns for stock
def compute_returns(df):
    df["log_return"] = df["Adj Close"].diff()
    df.dropna(inplace=True)
    return df


# If the output path does not exist, create it
output_path = "data/raw/SP500/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# For each ticker download the file, compute the returns, and save to parquet
for ticker in tqdm(tickers, desc="Downloading returns for tickers", unit=" tickers"):
    try:
        # Try downloading and processing the data
        df = load_data(ticker)
        df = compute_returns(df)
        file_name = ticker + "-returns.parquet"
        df.to_parquet(os.path.join(output_path, file_name))
    except Exception as e:
        # Handle any errors that occur
        print(f"Failed to process {ticker}: {e}")
        continue

print("Done downloading data.")
