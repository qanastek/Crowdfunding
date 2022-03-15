# Notes

## Data

### projects.csv

- Encoding is 'cp1252'.
- Missing values are simply 'None'.
- Some missing/null values in columns:
  - name
  - country
  - sex
- String columns do not benefit from standardization.
  - upper()
  - strip()
  - compress whitespaces
- Numerical Variables
  - id
  - age
  - goal
  - pledged
  - backers
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

## Processing
