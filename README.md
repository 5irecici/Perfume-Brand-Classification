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

# Data Preprocessing & Splitting Strategy:
The preprocessing pipeline handles missing records systematically (median for numericals, mode for categoricals) and maps text features to integers using LabelEncoder. To establish a reliable evaluation framework, the fully numeric feature space (X) and target array (y) are divided using a Stratified Train-Test Split (80/20). Stratification guarantees that the proportion of each perfume brand remains identical in both the training and testing phases, mitigating the risks associated with the remaining class imbalance.

# Feature Scaling & Leakage Prevention:
Before feeding the data into classification algorithms, the feature matrices were normalized using Standard Scaling. A strict boundary was maintained between the training and testing sets to prevent data leakage. The scaler was fitted exclusively on the training subset, ensuring that the test set remains completely unseen and acts as a true benchmark for model evaluation.

# Class Balancing Strategy (SMOTE):
Even within the Top 10 brands, the dataset exhibited varying frequencies. To ensure the classifiers learn the minority classes just as effectively as the majority ones, SMOTE was integrated into the pipeline. To strictly avoid data leakage and overly optimistic performance metrics, oversampling was performed exclusively on the training subset. The test set was left exactly as is, guaranteeing a robust, unbiased, and real-world evaluation of the trained models.

# Feature Selection & Correlation Analysis:
To optimize model performance and avoid the curse of dimensionality, a rigorous Pearson correlation analysis was conducted on the training subset. A heatmap was utilized to monitor feature collinearity, while target correlations were evaluated to identify the most predictive attributes. Features demonstrating an absolute correlation coefficient greater than the defined threshold (∣r∣>0.05) were flagged as significant inputs, guiding the feature selection process for the classifiers.

# Multicollinearity Prevention & Feature Elimination:
To construct a robust predictive model, it is vital to eliminate redundant features that carry overlapping information. A Pearson correlation matrix was computed to assess inter-feature relationships. An automated threshold mechanism was established to identify highly correlated independent variable pairs (∣r∣>0.90). By systematically dropping one feature from each highly correlated pair, the pipeline successfully mitigates the risk of multicollinearity. This dimensionality reduction step enhances both the computational efficiency and the interpretability of the final classifiers.
# Data Engineering & Preprocessing Architecture:
To prepare the raw dataset for classification, an end-to-end preprocessing pipeline was developed. After handling class imbalances by narrowing the scope to the Top 10 brands, missing data was systematically imputed. Identifier columns (URLs, specific perfume names) were strictly excluded from the feature space to prevent data leakage and overfitting. Following a stratified data split and standard scaling, a dynamic Pearson correlation filter was applied to eliminate multicollinearity (∣r∣>0.90). This rigorously cleaned, scaled, and non-collinear feature space is perfectly tailored to feed into advanced predictive models such as Random Forest and Gradient Boosting.
# Statistical Feature Selection (SelectKBest):
Following the removal of multicollinear attributes, an ANOVA F-test (SelectKBest) was integrated into the pipeline to perform rigorous feature selection. This algorithm computes the ANOVA F-value for each feature, identifying which inputs possess the most significant variance across the different brand categories. The feature space was successfully reduced to the Top 10 most informative predictors. This statistical reduction not only accelerates model training but also effectively minimizes the risk of overfitting by filtering out irrelevant noise.
# Feature Importance & Dimensionality Reduction:
Following the multicollinearity checks, a tree-based feature selection strategy was employed to further optimize the input space. A baseline Random Forest model was trained to compute the relative importance of each feature based on mean decrease in impurity. By isolating and visualizing the Top 10 most predictive attributes, the pipeline effectively filters out irrelevant noise. This strategic dimensionality reduction directly mitigates the risk of overfitting and ensures maximum computational efficiency during the final model training phase.
# Recursive Feature Elimination (RFE) Pipeline:
As the final step of the data engineering script, RFE was utilized to perform a rigorous feature selection. Using a Random Forest estimator, the algorithm recursively pruned the least significant attributes, considering complex feature interactions rather than just individual variances. This methodical reduction isolated the Top 10 fragrance features, ultimately mitigating overfitting, accelerating training times, and ensuring a highly interpretable and robust predictive model.
# Robust Feature Selection via Consensus (Voting):
To extract the absolute best predictors while aggressively reducing dimensionality, an ensemble feature selection architecture was designed. Instead of trusting a single metric, three diverse algorithms were deployed simultaneously:
SelectKBest (ANOVA F-Test): For statistical variance analysis.
Random Forest Importances: For non-linear, tree-based information gain.
Recursive Feature Elimination (RFE): For complex feature interactions.
A custom voting mechanism evaluated the results. Features were retained only if they were validated by at least two out of the three algorithms. This strict consensus protocol eliminated noisy variables, prevented overfitting, and yielded a lean, highly optimized feature matrix for model training.
# Dimensionality Reduction Strategy (PCA):
To evaluate the complexity and redundancy within the optimized feature space, a Principal Component Analysis (PCA) was conducted on the balanced training set. Through rigorous variance analysis (visualized via Scree and Cumulative Variance plots), the pipeline mathematically determined the optimal number of principal components required to preserve 90% and 95% of the total variance. This diagnostic step ensures that any future dimensionality reduction will successfully eliminate noise and speed up training times without sacrificing the predictive power needed to classify the perfume brands accurately.
# t-SNE Manifold Visualization & Class Separability:
To assess the non-linear separability of the Top 10 perfume brands, a t-SNE (t-Distributed Stochastic Neighbor Embedding) analysis was performed on the preprocessed feature matrix. By testing various perplexity hyperparameters (15, 30, 50), the algorithm successfully projected the complex, multi-dimensional fragrance signatures into a 2D topological map. The resulting visualizations revealed distinct clusters (islands) corresponding to easily separable brands, alongside denser, overlapping regions. These overlapping zones pinpointed specific fragrance profiles that share nearly identical top, middle, and base notes, forewarning areas where the classification models might encounter prediction ambiguities.
