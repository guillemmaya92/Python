# Libraries
# ==============================================
import polars as pl
import requests
from io import BytesIO
import os
from pathlib import Path

# Data Extraction
# ==============================================
# URL Parquet
url = "https://github.com/guillemmaya92/Analytics/raw/refs/heads/master/Data/wid.parquet"

# Create polars dataframe
response = requests.get(url)
df = pl.read_parquet(BytesIO(response.content))

# Data Manipulation
# ==============================================
# Filter specific measures
df = df.filter(
    (pl.col("year") >= 1980) &
    (pl.col("variable").is_in(["wpweali999", "wgweali999", "mpweali999", "mgweali999", "xlcusxi999"]))
)

# Pivot columns
df = df.pivot(
    values="value",
    index=["country", "year"],
    columns="variable"
)

# Add column wealth usd
df = df.with_columns([
    (pl.col("wpweali999")).alias("private_wealth_ratio"),
    (pl.col("wgweali999")).alias("public_wealth_ratio"),
    (pl.col("mgweali999") / pl.col("xlcusxi999")).alias("public_wealth"),
    (pl.col("mpweali999") / pl.col("xlcusxi999")).alias("private_wealth_usd"),
    ((pl.col("mgweali999") + pl.col("mpweali999")) / pl.col("xlcusxi999")).alias("total_wealth_usd")
]).select([
    "country",
    "year",
    "private_wealth_ratio",
    "public_wealth_ratio",
    "public_wealth",
    "private_wealth_usd",
    "total_wealth_usd"
])

# Clean dataframe
iso2_codes = [
    "AF", "AX", "AL", "DZ", "AS", "AD", "AO", "AI", "AQ", "AG", "AR", "AM", "AW", "AU", "AT", "AZ", "BS", "BH", "BD", "BB", "BY", "BE", "BZ", "BJ", "BM", "BT", "BO", "BQ", "BA", "BW",
    "BV", "BR", "IO", "BN", "BG", "BF", "BI", "CV", "KH", "CM", "CA", "KY", "CF", "TD", "CL", "CN", "CX", "CC", "CO", "KM", "CG", "CD", "CK", "CR", "CI", "HR", "CU", "CW", "CY", "CZ",
    "DK", "DJ", "DM", "DO", "EC", "EG", "SV", "GQ", "ER", "EE", "SZ", "ET", "FK", "FO", "FJ", "FI", "FR", "GF", "PF", "TF", "GA", "GM", "GE", "DE", "GH", "GI", "GR", "GL", "GD", "GP",
    "GU", "GT", "GG", "GN", "GW", "GY", "HT", "HM", "VA", "HN", "HK", "HU", "IS", "IN", "ID", "IR", "IQ", "IE", "IM", "IL", "IT", "JM", "JP", "JE", "JO", "KZ", "KE", "KI", "KP", "KR",
    "KW", "KG", "LA", "LV", "LB", "LS", "LR", "LY", "LI", "LT", "LU", "MO", "MG", "MW", "MY", "MV", "ML", "MT", "MH", "MQ", "MR", "MU", "YT", "MX", "FM", "MD", "MC", "MN", "ME", "MS",
    "MA", "MZ", "MM", "NA", "NR", "NP", "NL", "NC", "NZ", "NI", "NE", "NG", "NU", "NF", "MK", "MP", "NO", "OM", "PK", "PW", "PS", "PA", "PG", "PY", "PE", "PH", "PN", "PL", "PT", "PR",
    "QA", "RE", "RO", "RU", "RW", "BL", "SH", "KN", "LC", "MF", "PM", "VC", "WS", "SM", "ST", "SA", "SN", "RS", "SC", "SL", "SG", "SX", "SK", "SI", "SB", "SO", "ZA", "GS", "SS", "ES",
    "LK", "SD", "SR", "SJ", "SE", "CH", "SY", "TW", "TJ", "TZ", "TH", "TL", "TG", "TK", "TO", "TT", "TN", "TR", "TM", "TC", "TV", "UG", "UA", "AE", "GB", "US", "UM", "UY", "UZ", "VU",
    "VE", "VN", "VG", "VI", "WF", "EH", "YE", "ZM", "ZW"
]
df = df.filter(
    (pl.col("country").is_in(iso2_codes)) & 
    (pl.col("total_wealth_usd").is_not_null())
)

# Order dataframe
df = df.sort("total_wealth_usd", descending=True)

# Save Data
# ==============================================
download_folder = os.path.join(os.path.expanduser("~"), "Downloads")
filename = os.path.join(download_folder, "wealth.xlsx")
df.write_excel(filename)

# Show df
print(df)
