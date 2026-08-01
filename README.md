# Perfume-Brand-Classification
A machine learning project that classifies perfume brands based on fragrance notes, main accords, and user ratings. The dataset is optimized for the top 10 dominant brands to address the long-tail class imbalance and prevent underfitting.

# Dataset
A cleaned dataset containing 24,063 records and 18 features, including fragrance notes, main accords, release years, and user ratings. This file serves as the main input for the classification models before filtering for the top 10 dominant brands.

# Data Loading Note
The dataset is loaded using the latin-1 encoding and a semicolon (;) separator to ensure all fragrance names and European characters are parsed correctly without data loss.

# Data Quality & Missing Value Handling
Upon loading the dataset, a comprehensive missing value analysis is conducted. Attributes with missing records are identified and visualized as percentages. This step is crucial for determining which imputation methods (e.g., mode for categorical text like 'Notes', median for numerical columns) will be applied before training the classification models.

# Exploratory Data Analysis (EDA):
Initial data exploration is performed to uncover the underlying structure of the dataset. Categorical and numerical summaries are extracted, highlighting the most dominant brands and the general user rating distribution. This analysis plays a vital role in understanding the long-tail distribution of the brands before filtering the dataset for the classification models.

# Addressing Class Imbalance and "Long-Tail" Problem:
Machine learning algorithms often fail to generalize when faced with classes that have very few samples. By analyzing the frequency distribution of the Brand target variable, a massive long-tail distribution was identified. To build statistically meaningful decision boundaries, the dataset was filtered to include only the top 10 most frequent brands. This strategic reduction ensures that the models focus on classes with enough depth and representation.

# Long-Tail Visualization & Noise Reduction:
An analysis of the Brand target variable revealed a classic power-law distribution. The generated area chart clearly shows a massive "long tail" of underrepresented classes. Training a model on over 1,000 imbalanced classes introduces significant noise and leads to underfitting. Therefore, the dataset was systematically reduced to the Top 10 brands. This noise reduction step provides a statistically stable foundation for training high-accuracy classification algorithms.

# Target Variable Profiling & Imbalance Strategy:
Post-filtering, the dataset comprises the 10 most prominent perfume brands. A detailed distribution analysis (utilizing customized bar and pie charts) was conducted on this refined subset. The calculated class imbalance ratio highlighted the underlying distribution gaps, guiding the strategic decision to incorporate class balancing techniques (e.g., SMOTE) into the subsequent machine learning pipeline to ensure fair and unbiased predictions across all brands.

# Numerical Distribution & Feature Variance:
To understand the underlying structure of the numerical data, a grouped distribution analysis was performed on the Top 10 brands. By plotting histograms for metrics such as User Ratings, Vote Counts, and Release Years, we observed class-specific variances and skewness. This granular level of EDA confirmed the need for appropriate feature scaling and standardization techniques to ensure the classification algorithms treat all numerical inputs fairly.

# Data Preprocessing & Encoding Pipeline:
To ensure compatibility with machine learning classifiers, a robust preprocessing pipeline was implemented. Missing data was handled using appropriate imputation strategies based on data types (median for numericals, mode for categoricals). Additionally, since algorithms require numerical inputs, all categorical text features and the target variable were mapped to integer values using LabelEncoder. This resulted in a fully optimized feature space (X) and target array (y).
