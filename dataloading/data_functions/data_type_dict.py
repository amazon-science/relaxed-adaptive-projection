NUMERIC = "numerical"
CATEGORICAL = "categorical"
FEATURE_TYPES = {
    "AGEP": NUMERIC,
    "COW": CATEGORICAL,  # Class of worker
    "SCHL": CATEGORICAL,  # Educational attainment
    "MAR": CATEGORICAL,  # Marital status
    "OCCP": CATEGORICAL,  # Occupation recode for 2018 and later based on 2018 OCC codes
    "POBP": CATEGORICAL,  # Place of birth
    "RELP": CATEGORICAL,
    "WKHP": NUMERIC,  # Usual hours worked per week past 12 months
    "SEX": CATEGORICAL,
    "RAC1P": CATEGORICAL,  # Recoded detailed race code
    "PINCP": NUMERIC,  # Total person's income
    "WAOB": CATEGORICAL,  # World area of birth
    "FOCCP": CATEGORICAL,  # Occupation allocation flag (binary)
    "PWGTP": NUMERIC,  # Person weight
    "INTP": NUMERIC,  # Interest, dividends, and net rental income past 12 months (signed, use ADJINC to adjust to constant dollars
    "JWMNP": NUMERIC,  # Travel time to work
    "JWRIP": NUMERIC,  # Vehicle occupancy
    "PAP": NUMERIC,  # Public assistance income past 12 months (use ADJINC to adjust to constant dollars)
    "SEMP": NUMERIC,  # Self-employment income past 12 months (signed, use ADJINC to adjust SEMP to constant dollars)
    "WAGP": NUMERIC,  # Wages or salary income past 12 months (use ADJINC to adjust WAGP to constant dollars)
    "DIS": CATEGORICAL,  # Disability recode (binary)
    "ESP": CATEGORICAL,  # Employment status of parents
    "MIG": CATEGORICAL,  # Mobility status (lived here 1 year ago)
    "PUMA": CATEGORICAL,  # Public use microdata area code (PUMA)
    "ST": CATEGORICAL,  # State code based on 2010 Census definition
    "CIT": CATEGORICAL,  # Citizenship status
    "JWTR": CATEGORICAL,  # Means of transportation to work
    "POWPUMA": CATEGORICAL,  # Place of work PUMA based on 2010 Census definitions
    "POVPIP": NUMERIC,  # Income-to-poverty ratio recode
    "MIL": CATEGORICAL,  # Served September 2001 or later
    "ANC": CATEGORICAL,  # Ancestry recode
    "NATIVITY": CATEGORICAL,  # Nativity
    "DEAR": CATEGORICAL,  # Hearing difficulty
    "DEYE": CATEGORICAL,  # Vision difficulty
    "DREM": CATEGORICAL,  #  Cognitive difficulty
    "GCL": CATEGORICAL,  # Grandparents living with grandchildren
    "ESR": CATEGORICAL,  # Employment status recode
    "FER": CATEGORICAL,  # Gave birth to child within the past 12 months b .N/A (less than 15 years/greater than 50 years/male) 1 .Yes 2 .No
}
