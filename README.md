# Mammal Taxonomy Gradient Boosting Classifier

## Description

In this project, I create a classifier to predict a mammal's taxonomic order based on its ecological, geographic, and life-history traits. The classifier is trained on data from PanTHERIA, a comprehensive species-level database containing trait data for all known extant and recently-extinct mammals. Exploratory data analysis is conducted with Pandas, while preprocessing, model selection, and hyperparameter tuning are performed in Scikit-learn. After numerous rounds of cross validation within the training set, the best-performing model was found to be a histogram-based gradient boosting classifier with a learning rate of 0.01, a minimum leaf size of 50, and a maximum of 2000 iterations of the boosting process. As part of the classification pipeline, this model also:
- transformed both positively-skewed and negatively-skewed numeric features 
- standardized all numeric features
- leveraged the native support for missing values and categorical features offered by Scikit-learn's ```HistGradientBoostingClassifier```
- undersampled the majority order&mdash;Rodentia&mdash;to one-fourth its prevalence in the training set

Ultimately, when evaluated on the test set, the classifier achieved a balanced accuracy score of 0.785, with all but two of the ten considered orders attaining a recall score of at least 0.75. While not a substitute for methodical phylogenetic reconstruction, this model helps illuminate some of the central affinities within the mammalian class; it is no coincidence, for example, that the classifier was particularly prone to characterizing cetaceans as artiodactyls&mdash;a striking reflection of their close evolutionary kinship.

## Dependencies

- NumPy
- Pandas
- Scikit-learn
- Imbalanced-learn
- SciPy
- Matplotlib
- Seaborn

For a full list of dependencies&mdash;both direct and transitive&mdash;please refer to the provided requirements.txt file.

## Usage

An annotated walk-through of the entire process can be found at  ```/model/Mammal Taxonomy Gradient Boosting Classifier.ipynb```. The final model itself is stored as a byte stream at ```model/classifier.pkl```; these bytes can be converted into a Python object with Python's ```pickle``` module. The raw data, though also accessible via [this link](https://esapubs.org/archive/ecol/E090/184/PanTHERIA_1-0_WR05_Aug2008.txt), is saved at ```/data/pantheria.txt```, while cross validation output is stored in ```/cv_output```. For full variable definitions and in-depth discussion of how these variables were derived, please see PanTHERIA's [metadata file](https://esapubs.org/archive/ecol/E090/184/metadata.htm).

## Summary

Trait data for thousands of mammalian species were obtained from PanTHERIA, a comprehensive species-level database of mammalian traits. These data were imported into the file, received preliminary cleaning (e.g., inserting estimates into empirically-derived columns, filtering for the 10 most-frequent orders), and were subsequently partitioned into a train set and test set in a stratified fashion. Exploratory data analysis with the train set followed in Pandas, dropping features with insufficient data and assessing the visual and statistical impacts of missing-value imputation, skewness transformations, standardization, dimensionality reduction via PCA, and one-hot encoding. Based on these transformations, a preprocessing pipeline was formulated in Scikit-learn, where it was then tested using grid-search cross validation (CV) with a variety of different classifier types&mdash;decision trees, random forests, gradient boosting classifiers (GBCs), and support vector classifiers with different kernels. The best-performing classifiers from this round of CV&mdash;as judged by mean balanced accuracy across 5 stratified splits of the training data&mdash;were random forests and gradient boosting classifiers; these classifiers performed optimally when they did not reduce the original dimensionality of the numeric features. Having narrowed down the classifiers to these two types, random forests and GBCs were each run through 200 iterations of CV with randomized hyperparameter settings; random forests varied over metrics such as number of constituent trees and the minimum number of samples to split an internal node, while GBCs varied across learning rate, minimum leaf size, L2 regularization, and maximum number of boosting iterations. Both types of models were also allowed to vary over the imputing method for numeric features and the precise skewness thresholds around which to apply log transformations and square-root transformations.

Across these combined 400 rounds of cross validation, GBCs outperformed random forests both overall and at the extreme upper edge of performance, with top-performers making use of a mean imputation strategy and a maximum of 500 or more boosting iterations. It was then discovered that the top classifier from all these 400 rounds&mdash;a GBC with a learning rate of 0.01, a minimum leaf size of 50, and a maximum of 2000 iterations of the boosting process&mdash;could be improved even further by leaving missing values untouched and by specifying categorical features by index&mdash;eliminating the need for one-hot encoding. Examining this model's precise prediction errors, it was notable that predictions of Rodentia could be found in every single order&mdash;particularly in lagomorphs and soricomorphs, which obtained mediocre recall scores. Given the training data's great imbalance toward Rodentia, two resampling techniques were evaluated: in the first case, the model randomly undersampled the majority order&mdash;Rodentia&mdash;to one-fourth its prevalence in the training set, while in the second case, the model randomly oversampled the number of instances of all non-majority orders (non-Rodentia orders) up to the number of Rodentia instances in the training set. Both of these techniques led to noteworthy performance improvements compared to the non-resampled model, but the undersampled model marginally outperformed the oversampled model. After determining that this undersampled model did not perform better with more regularization, it was taken to final evaluation on the test set, where it achieved a balanced accuracy of 0.785 and a recall score of at least 0.75 on eight out of the ten orders considered. The classifier was then trained on the entire pre-partitioned dataset and saved down as a byte stream for future use.

## Data Sources and Citations

All data come from the PanTHERIA database (Jones et al., 2009), using the taxonomy set forth by Wilson and Reeder in 2005 (Wilson & Reeder, 2005). Raw data can be found [here](https://esapubs.org/archive/ecol/E090/184/PanTHERIA_1-0_WR05_Aug2008.txt), with accompanying metadata [here](https://esapubs.org/archive/ecol/E090/184/metadata.htm). Full citations are below.

Jones, K. E., Bielby, J., Cardillo, M., Fritz, S. A., O'Dell, J., Orme, C. D., Safi, K., Sechrest, W., Boakes, E. H., Carbone, C., Connolly, C., Cutts, M. J., Foster, J. K., Grenyer, R., Habib, M., Plaster, C. A., Price, S. A., Rigby, E. A., Rist, J., . . . Purvis, A. (2009). PanTHERIA: a species-level database of life history, ecology, and geography of extant and recently extinct mammals. Ecology, 90(9), 2648. https://doi.org/10.1890/08-1494.1

Wilson, D. E., & Reeder, D. M. (2005). Mammal species of the world: a taxonomic and geographic reference. JHU press.