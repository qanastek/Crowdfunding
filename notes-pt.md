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

## Processing
