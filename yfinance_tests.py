import yfinance as yf

msft = yf.Ticker("MSFT")

hist = msft.history(period="max")
print(hist)