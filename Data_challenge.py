import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
cover_df = pd.read_csv("/Users/sahil/Downloads/CREMP_CSV_files/CREMP_Pcover_2023_StonyCoralSpecies.csv")
species_df = pd.read_csv("/Users/sahil/Downloads/CREMP_CSV_files/CREMP_SCOR_Summaries_2023_Counts.csv")
temp_df = pd.read_csv("/Users/sahil/Downloads/CREMP_CSV_files/CREMP_Temperatures_2023.csv")

# Preview each dataset
print("=== Coral Cover Data ===")
print(cover_df.head(), "\n")

print("=== Species Richness Data ===")
print(species_df.head(), "\n")

print("=== Temperature Data ===")
print(temp_df.head(), "\n")

#missing vals

print("=== Missing Values: Coral Cover ===")
print(cover_df.isnull().sum())

print("=== Missing Values: Species Richness ===")
print(species_df.isnull().sum())

print("=== Missing Values: Temperature ===")
print(temp_df.isnull().sum())


cover_df['Date'] = pd.to_datetime(cover_df['Date'])
species_df['Date'] = pd.to_datetime(species_df['Date'])


# Get all coral species columns (start after 'points')
species_columns = cover_df.columns[cover_df.columns.get_loc('points') + 1:]

# Create a new column for total coral cover
cover_df['Total'] = cover_df[species_columns].sum(axis=1)

 # === OBJECTIVE: Analyze long-term trends in stony coral percent cover ===
# Now group by year and calculate average percent cover
cover_by_year = cover_df.groupby('Year')['Total'].mean()
cover_by_year.plot(kind='line', marker='o', title='Average Stony Coral Percent Cover Over Years')
plt.ylabel('% Coral Cover')
plt.xlabel('Year')
plt.grid(True)
plt.show()

# === OBJECTIVE: Identify and interpret trends in species richness of stony corals ===
# === Species Richness Trends ===

# Sum all species presence values per row (presence is counted if value > 0)
species_presence = species_df.iloc[:, 7:] > 0  # skip first 7 meta columns
species_df['Richness'] = species_presence.sum(axis=1)

# Group by year and average richness
richness_by_year = species_df.groupby('Year')['Richness'].mean()

# Plot species richness over time
richness_by_year.plot(kind='line', marker='o', title='Average Species Richness Over Years', color='green')
plt.ylabel('Species Richness')
plt.xlabel('Year')
plt.grid(True)
plt.show()

# === OBJECTIVE: Examine correlation between coral cover and water temperature ===
# === Coral Cover vs Temperature Correlation ===

# Compute average temperature per year
avg_temp_by_year = temp_df.groupby('Year')['TempC'].mean()

# Merge with average coral cover
combined_df = pd.DataFrame({
    'AvgTemp': avg_temp_by_year,
    'AvgCoralCover': cover_by_year
}).dropna()

# Plot relationship between temperature and coral cover
sns.regplot(data=combined_df, x='AvgTemp', y='AvgCoralCover')
plt.title("Coral Cover vs Average Temperature")
plt.xlabel("Average Temperature (°C)")
plt.ylabel("Average Coral Cover (%)")
plt.grid(True)
plt.show()

# Show correlation coefficient
print("Correlation matrix:\n", combined_df.corr())

# === OBJECTIVE: Evaluate spatial patterns in coral cover across subregions ===
# === Subregion-wise Coral Cover Trends ===

# Compute average coral cover per subregion per year
subregion_cover = cover_df.groupby(['Year', 'Subregion'])['Total'].mean().reset_index()

# Plot trends
plt.figure(figsize=(12, 6))
sns.lineplot(data=subregion_cover, x='Year', y='Total', hue='Subregion', marker='o')
plt.title("Average Coral Cover by Subregion Over Years")
plt.ylabel('% Coral Cover')
plt.xlabel('Year')
plt.grid(True)
plt.legend(title='Subregion')
plt.show()

# === OBJECTIVE: Evaluate spatial patterns in species richness across subregions ===
# === Subregion-wise Species Richness Trends ===
subregion_richness = species_df.groupby(['Year', 'Subregion'])['Richness'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=subregion_richness, x='Year', y='Richness', hue='Subregion', marker='o')
plt.title("Average Species Richness by Subregion Over Years")
plt.ylabel('Species Richness')
plt.xlabel('Year')
plt.grid(True)
plt.legend(title='Subregion')
plt.show()

# === OBJECTIVE: Model future coral cover trends to anticipate declines ===
# === Forecasting Coral Cover Using Linear Regression ===
from sklearn.linear_model import LinearRegression
import numpy as np

# Prepare data
X_years = cover_by_year.index.values.reshape(-1, 1)
y_cover = cover_by_year.values

# Fit model
model = LinearRegression()
model.fit(X_years, y_cover)

# Predict for existing + future years
future_years = np.arange(cover_by_year.index.min(), cover_by_year.index.max() + 10).reshape(-1, 1)
predicted_cover = model.predict(future_years)

# Plot actual + forecasted coral cover
plt.figure(figsize=(10, 5))
plt.plot(X_years, y_cover, 'o-', label='Actual Coral Cover')
plt.plot(future_years, predicted_cover, 'r--', label='Forecasted Coral Cover')
plt.title('Coral Cover Forecast (Linear Regression)')
plt.xlabel('Year')
plt.ylabel('% Coral Cover')
plt.legend()
plt.grid(True)
plt.show()

# === OBJECTIVE: Statistically test the relationship between temperature and coral cover ===
# === Hypothesis Testing: Temperature Impact on Cover ===
from scipy.stats import pearsonr

corr_coeff, p_value = pearsonr(combined_df['AvgTemp'], combined_df['AvgCoralCover'])
print(f"Pearson correlation: {corr_coeff:.2f}, p-value: {p_value:.4f}")
if p_value < 0.05:
    print("→ Statistically significant relationship between temperature and coral cover.")
else:
    print("→ No statistically significant relationship between temperature and coral cover.")

# === OBJECTIVE: Analyze trends in stony coral cover at station level ===
# === Coral Cover by Station Over Time ===
station_cover = cover_df.groupby(['Year', 'SiteID'])['Total'].mean().reset_index()

plt.figure(figsize=(14, 6))
sns.lineplot(data=station_cover, x='Year', y='Total', hue='SiteID', legend=False)
plt.title("Stony Coral Percent Cover by Station Over Time")
plt.xlabel("Year")
plt.ylabel("Percent Cover")
plt.grid(True)
plt.show()

# === OBJECTIVE: Analyze species richness trends at station level ===
# === Species Richness by Station Over Time ===
station_richness = species_df.groupby(['Year', 'SiteID'])['Richness'].mean().reset_index()

plt.figure(figsize=(14, 6))
sns.lineplot(data=station_richness, x='Year', y='Richness', hue='SiteID', legend=False)
plt.title("Stony Coral Species Richness by Station Over Time")
plt.xlabel("Year")
plt.ylabel("Species Richness")
plt.grid(True)
plt.show()

# === OBJECTIVE: Examine octocoral density trends over time and across stations ===
# === Octocoral Density Analysis ===
octo_df = pd.read_csv("/Users/sahil/Downloads/CREMP_CSV_files/CREMP_OCTO_Summaries_2023_Density.csv")

# Convert date if necessary (not needed here as year is already available)
if 'Date' in octo_df.columns:
    octo_df['Date'] = pd.to_datetime(octo_df['Date'], errors='coerce')
    octo_df['Year'] = octo_df['Date'].dt.year

# Ensure the 'Year' column exists and is integer
octo_df['Year'] = octo_df['Year'].astype(int)

# Group by year and site to analyze Total_Octocorals
octo_density_trend = octo_df.groupby(['Year', 'SiteID'])['Total_Octocorals'].mean().reset_index()

# Plot
plt.figure(figsize=(14, 6))
sns.lineplot(data=octo_density_trend, x='Year', y='Total_Octocorals', hue='SiteID', legend=False)
plt.title("Octocoral Density by Station Over Time")
plt.xlabel("Year")
plt.ylabel("Total Octocoral Density")
plt.grid(True)
plt.show()

# === OBJECTIVE: Determine site-wise variation in living tissue area ===
# === Living Tissue Area Variability ===
# Assumes 'Scleractinia' column is related to living tissue area
if 'Scleractinia' in cover_df.columns:
    site_tissue_area = cover_df.groupby('SiteID')['Scleractinia'].mean().reset_index()
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=cover_df, x='SiteID', y='Scleractinia')
    plt.title("Living Tissue Area (Scleractinia) Across Sites")
    plt.xlabel("Site ID")
    plt.ylabel("Mean Living Tissue Area")
    plt.xticks(rotation=90)
    plt.show()

# === OBJECTIVE: Assess spatial and temporal variation in dominant coral species ===
# === Coral Species Distribution Over Time and Space ===
top_species = cover_df[species_columns].mean().sort_values(ascending=False).head(5).index.tolist()
melted = cover_df.melt(id_vars=['Year', 'Subregion'], value_vars=top_species, var_name='Species', value_name='Cover')

plt.figure(figsize=(12, 6))
sns.lineplot(data=melted, x='Year', y='Cover', hue='Species')
plt.title("Top 5 Coral Species Cover Over Time")
plt.xlabel("Year")
plt.ylabel("Percent Cover")
plt.grid(True)
plt.show()

# === OBJECTIVE: Assess relationship between coral cover and species richness ===
# === Coral Cover vs Richness Relationship ===
site_summary = cover_df.groupby('SiteID')['Total'].mean().reset_index()
richness_summary = species_df.groupby('SiteID')['Richness'].mean().reset_index()
coral_richness_df = pd.merge(site_summary, richness_summary, on='SiteID').dropna()

sns.scatterplot(data=coral_richness_df, x='Total', y='Richness')
plt.title("Relationship Between Coral Cover and Species Richness")
plt.xlabel("Average Coral Cover (%)")
plt.ylabel("Average Species Richness")
plt.grid(True)
plt.show()

# === Future Outlook Forecast for Coral Cover ===
# (Already implemented in earlier forecast block)

# === OBJECTIVE: Identify key influencing species and detect early indicators of decline ===
# === Identify Key Factors Affecting Coral Health ===
corr_matrix = cover_df[species_columns.tolist() + ['Total']].corr()
print(corr_matrix['Total'].sort_values(ascending=False).head())

# Additional early indicators (e.g., slope changes, sudden drops) can be derived from yearly differences
cover_diff = cover_by_year.diff()
significant_drops = cover_diff[cover_diff < -1.0]
print("Years with significant declines in coral cover:")
print(significant_drops)