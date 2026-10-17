import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------------------------------
# 1. LOAD CLEANED DATA
# ---------------------------------------------------

file = "EVChargingStationUsage_cleaned.csv"

df = pd.read_csv(file, low_memory=False)

print("Dataset loaded successfully")
print("Shape:", df.shape)


# ---------------------------------------------------
# 2. CREATE OUTPUT FOLDER
# ---------------------------------------------------

os.makedirs("eda_outputs", exist_ok=True)


# ---------------------------------------------------
# 3. BASIC INFORMATION
# ---------------------------------------------------

print("\n========== DATASET INFORMATION ==========")

print("\nRows and Columns:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)


# ---------------------------------------------------
# 4. MISSING VALUES
# ---------------------------------------------------

print("\n========== MISSING VALUES ==========")

missing = df.isnull().sum()

missing = missing[missing > 0].sort_values(
    ascending=False
)

print(missing)


# ---------------------------------------------------
# 5. DUPLICATES
# ---------------------------------------------------

print("\n========== DUPLICATES ==========")

print("Duplicate rows:", df.duplicated().sum())


# ---------------------------------------------------
# 6. NUMERICAL SUMMARY
# ---------------------------------------------------

print("\n========== NUMERICAL SUMMARY ==========")

print(
    df.describe()
)


# ---------------------------------------------------
# 7. CATEGORICAL SUMMARY
# ---------------------------------------------------

print("\n========== CATEGORICAL SUMMARY ==========")

print("\nPort Type:")
print(df["Port Type"].value_counts())

print("\nPlug Type:")
print(df["Plug Type"].value_counts())

print("\nEnded By:")
print(df["Ended By"].value_counts().head(15))

print("\nModel Number:")
print(df["Model Number"].value_counts())


# ---------------------------------------------------
# 8. TOP 10 CHARGING STATIONS
# ---------------------------------------------------

top_stations = (
    df["Station Name"]
    .value_counts()
    .head(10)
)

print("\n========== TOP 10 STATIONS ==========")
print(top_stations)


# ---------------------------------------------------
# 9. TOTAL ENERGY
# ---------------------------------------------------

total_energy = df["Energy (kWh)"].sum()

print("\nTotal Energy Consumed:")
print(round(total_energy, 2), "kWh")


# ---------------------------------------------------
# 10. AVERAGE ENERGY PER SESSION
# ---------------------------------------------------

average_energy = df["Energy (kWh)"].mean()

print("\nAverage Energy Per Charging Session:")
print(round(average_energy, 2), "kWh")


# ---------------------------------------------------
# 11. TOTAL FEE
# ---------------------------------------------------

total_fee = df["Fee"].sum()

print("\nTotal Fee:")
print(round(total_fee, 2))


# ---------------------------------------------------
# 12. ENERGY DISTRIBUTION
# ---------------------------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    df["Energy (kWh)"].dropna(),
    bins=50,
    kde=True
)

plt.title("Distribution of Energy Consumption")
plt.xlabel("Energy (kWh)")
plt.ylabel("Number of Charging Sessions")

plt.tight_layout()

plt.savefig(
    "eda_outputs/energy_distribution.png"
)

plt.show()


# ---------------------------------------------------
# 13. TOP 10 STATIONS
# ---------------------------------------------------

plt.figure(figsize=(12, 6))

top_stations.sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Charging Stations")
plt.xlabel("Number of Charging Sessions")
plt.ylabel("Station")

plt.tight_layout()

plt.savefig(
    "eda_outputs/top_10_stations.png"
)

plt.show()


# ---------------------------------------------------
# 14. PORT TYPE DISTRIBUTION
# ---------------------------------------------------

plt.figure(figsize=(8, 6))

df["Port Type"].value_counts().plot(
    kind="bar"
)

plt.title("Charging Port Type Distribution")
plt.xlabel("Port Type")
plt.ylabel("Number of Sessions")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "eda_outputs/port_type_distribution.png"
)

plt.show()


# ---------------------------------------------------
# 15. PLUG TYPE DISTRIBUTION
# ---------------------------------------------------

plt.figure(figsize=(8, 6))

df["Plug Type"].value_counts().plot(
    kind="bar"
)

plt.title("Plug Type Distribution")
plt.xlabel("Plug Type")
plt.ylabel("Number of Sessions")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "eda_outputs/plug_type_distribution.png"
)

plt.show()


# ---------------------------------------------------
# 16. ENERGY BY PORT TYPE
# ---------------------------------------------------

energy_by_port = (
    df.groupby("Port Type")["Energy (kWh)"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(8, 6))

energy_by_port.plot(
    kind="bar"
)

plt.title("Average Energy Consumption by Port Type")
plt.xlabel("Port Type")
plt.ylabel("Average Energy (kWh)")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "eda_outputs/energy_by_port_type.png"
)

plt.show()


# ---------------------------------------------------
# 17. CHARGING SESSIONS BY HOUR
# ---------------------------------------------------

hourly_sessions = (
    df["Hour"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(12, 6))

hourly_sessions.plot(
    kind="line",
    marker="o"
)

plt.title("Charging Sessions by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Sessions")

plt.xticks(range(0, 24))

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "eda_outputs/sessions_by_hour.png"
)

plt.show()


# ---------------------------------------------------
# 18. CHARGING SESSIONS BY DAY OF WEEK
# ---------------------------------------------------

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

day_sessions = (
    df["Day of Week"]
    .value_counts()
    .reindex(day_order)
)

plt.figure(figsize=(10, 6))

day_sessions.plot(
    kind="bar"
)

plt.title("Charging Sessions by Day of Week")
plt.xlabel("Day")
plt.ylabel("Number of Sessions")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "eda_outputs/sessions_by_day.png"
)

plt.show()


# ---------------------------------------------------
# 19. MONTHLY ENERGY CONSUMPTION
# ---------------------------------------------------

monthly_energy = (
    df.groupby(["Year", "Month"])["Energy (kWh)"]
    .sum()
)

monthly_energy = monthly_energy.reset_index()

monthly_energy["Date"] = pd.to_datetime(
    monthly_energy["Year"].astype(str)
    + "-"
    + monthly_energy["Month"].astype(str)
    + "-01"
)

plt.figure(figsize=(14, 6))

plt.plot(
    monthly_energy["Date"],
    monthly_energy["Energy (kWh)"],
    marker="o"
)

plt.title("Monthly Energy Consumption")
plt.xlabel("Date")
plt.ylabel("Energy (kWh)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "eda_outputs/monthly_energy.png"
)

plt.show()


# ---------------------------------------------------
# 20. CORRELATION HEATMAP
# ---------------------------------------------------

numeric_columns = [
    "Energy (kWh)",
    "GHG Savings (kg)",
    "Gasoline Savings (gallons)",
    "Fee",
    "Total Duration Minutes",
    "Charging Time Minutes"
]

correlation = df[numeric_columns].corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "eda_outputs/correlation_heatmap.png"
)

plt.show()


# ---------------------------------------------------
# 21. ENERGY VS CHARGING TIME
# ---------------------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df.sample(
        min(10000, len(df)),
        random_state=42
    ),
    x="Charging Time Minutes",
    y="Energy (kWh)"
)

plt.title("Energy Consumption vs Charging Time")
plt.xlabel("Charging Time (Minutes)")
plt.ylabel("Energy (kWh)")

plt.tight_layout()

plt.savefig(
    "eda_outputs/energy_vs_charging_time.png"
)

plt.show()


# ---------------------------------------------------
# 22. FINAL EDA SUMMARY
# ---------------------------------------------------

print("\n======================================")
print("EDA COMPLETED SUCCESSFULLY")
print("======================================")

print("Total charging sessions:", len(df))

print(
    "Total energy:",
    round(df["Energy (kWh)"].sum(), 2),
    "kWh"
)

print(
    "Average energy/session:",
    round(df["Energy (kWh)"].mean(), 2),
    "kWh"
)

print(
    "Total fees:",
    round(df["Fee"].sum(), 2)
)

print(
    "Average charging time:",
    round(df["Charging Time Minutes"].mean(), 2),
    "minutes"
)

print("\nAll graphs saved inside:")
print("eda_outputs/")