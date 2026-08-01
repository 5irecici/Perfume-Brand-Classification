# Perfume-Brand-Classification
A machine learning project that classifies perfume brands based on fragrance notes, main accords, and user ratings. The dataset is optimized for the top 10 dominant brands to address the long-tail class imbalance and prevent underfitting.

# Dataset
A cleaned dataset containing 24,063 records and 18 features, including fragrance notes, main accords, release years, and user ratings. This file serves as the main input for the classification models before filtering for the top 10 dominant brands.

# Data Loading Note
The dataset is loaded using the latin-1 encoding and a semicolon (;) separator to ensure all fragrance names and European characters are parsed correctly without data loss.

# Data Quality & Missing Value Handling
Upon loading the dataset, a comprehensive missing value analysis is conducted. Attributes with missing records are identified and visualized as percentages. This step is crucial for determining which imputation methods (e.g., mode for categorical text like 'Notes', median for numerical columns) will be applied before training the classification models.
