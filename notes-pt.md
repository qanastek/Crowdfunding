# Notes

## Data

### projects.csv

- Encoding is 'cp1252'.
- Missing values are simply 'None'.
- Missing Values
  - name (count: 4)
  - country (count: 3797)
  - sex (count: 8908)
- Outliers
  - age (count: , reason: Unix epoch default date value)
  - goal (count: , reason: Unix epoch default date value)
  - pledged (count: , reason: Unix epoch default date value)
  - backers (count: , reason: Unix epoch default date value)
  - start_date (count: , reason: Unix epoch default date value)
    - See <https://en.wikipedia.org/wiki/Unix_time>
- String columns do not benefit from standardization.
  - upper()
  - strip()
  - compress whitespaces
- Numerical Variables
  - id
  - age
  - goal (même fonctionnalité que 'pledge')
  - pledged (hors de notre contrôle, ignore)
  - backers (hors de notre contrôle, ignore)
- Categorical Variables
  - name (unique: 297872)
  - category (unique: 15)
  - subcategory (unique: 158)
  - country (unique: 18)
  - sex (unique: 2)
  - start_date (unique: 300179)
  - end_date (unique: 274233)
  - currency (unique: 10)
  - state (unique: 6)
- Nominal vs Ordinal?
- Label Variable
  - state
- Goal
  - Classification binaire (Success ou Fail)
    - Transformer les labels (state) en binaire en groupant ou supprimant les valeurs.
      - Group 'failed' et 'canceled'
      - Remove 'live', 'suspended' et 'undefined'

## Visualisation

### TODO

- countplots for each categorical variables by sets of 10 values (max colors in color palette)
- FacetPlot for each combination of categorical and numerical variable.

### References

- <https://www.kaggle.com/abhishekmamidi/everything-you-can-do-with-seaborn>
- <https://towardsdatascience.com/pair-plot-and-pairgrid-in-details-f782975032ea>
- <http://seaborn.pydata.org/tutorial/categorical.html>
- <https://seaborn.pydata.org/generated/seaborn.PairGrid.html>
- <https://seaborn.pydata.org/tutorial/color_palettes.html>
- <https://seaborn.pydata.org/tutorial/axis_grids.html>

## Data Cleaning (in general, NETTOYAGE)

- Transform values
  - Calculate 'duration' from 'start_date' and 'end_date'.
  - Normalize 'currency' to USD in 'goal' for more accurate comparisons.
  - Reduce problem to a binary classification (label => Success 0 or 1)
    - Group 'state' values of 'failed' and 'canceled' together.
    - Remove 'state' values of 'live', 'suspended' and 'undefined'.
- Remove outliers
  - Remove entries with invalid dates set to Unix epoch (January 1, 1970).
- Impute missing values
  - Impute missing numerical values via mean, median or constant 'UNK' value.
    - N/A
  - Impute missing categorical values via most frequent term or constant 'UNK' value.
    - sex
  - Impute missing 'country' values from 'currency' used.
- Remove useless columns
  - Variables that are identifiers.
    - id
    - name
  - Variables that are dates.
    - start_date
    - end_date
  - Variables that are outside of our control if we try to recreate a success.
    - pledged
    - backers

## Data Preprocessing (for training, RECODAGE + PRÉTRAITEMENT)

- One-Hot encode the categorical variables.
  - category
  - subcategory
  - country
  - sex
- Normalize the sample sizes for each 'state'. (upsampling vs downsampling)
- Compute term frequencies (TF-IDF) for 'names'.
