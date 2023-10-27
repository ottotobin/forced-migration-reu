import pandas as pd
import matplotlib.pyplot as plt

with open("combined_4_27_to_7_12.csv","r") as f:
    df = pd.read_csv(f)

df["Leaving IDP"] = df["Leaving IDP"].fillna(0)
df["Date"] = pd.to_datetime(df["Date"])

appends = []
state_val = {}
for date in sorted(df["Date"].unique()):
    for state in sorted(df["State of Displacement"].unique()):
        if state not in state_val:
            state_val[state] = 0

        row_exists = (df['Date'] == date) & (df['State of Displacement'] == state)
        if row_exists.any():
            state_val[state] = df[row_exists]["Arriving IDP"]
        else:
            appends.append({
                "Date":date,
                "State of Displacement":state,
                "Arriving IDP":state_val[state]
            })

print(appends)

# exit()

# df = df[df["State of Displacement"] == "North Darfur"]

# Pivot the DataFrame to have countries as columns and dates as the index
pivoted = df.pivot(index='Date', columns='State of Displacement', values='Leaving IDP')

# Create a line plot with all countries on the same plot
fig, ax = plt.subplots(figsize=(10, 6))

for country in pivoted.columns:
    ax.plot(pivoted.index, pivoted[country], label=country)

# Customize the plot
ax.set_xlabel('Date')
ax.set_ylabel('Count')
ax.set_title('Count vs. Date')
ax.legend()

# Show the plot
plt.show()