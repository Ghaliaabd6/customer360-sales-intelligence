import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


FILES = {
    "customers": "customers.csv",
    "products": "products.csv",
    "sales_representatives": "sales_representatives.csv",
    "sales": "sales.csv",
    "leads": "leads.csv",
    "opportunities": "opportunities.csv",
    "campaigns": "campaigns.csv",
    "campaign_interactions": "campaign_interactions.csv",
    "support_tickets": "support_tickets.csv",
}


# ============================================================
# LOAD RAW DATA
# ============================================================

dataframes = {}

for table_name, filename in FILES.items():

    path = RAW_DIR / filename

    if path.exists():

        dataframes[table_name] = pd.read_csv(
            path,
            low_memory=False
        )

        print(
            f"[OK] Loaded {table_name}: "
            f"{len(dataframes[table_name]):,} rows"
        )

    else:
        print(f"[MISSING] {path}")


# ============================================================
# GENERIC CLEANING FUNCTIONS
# ============================================================

def clean_text_columns(df):

    """
    Removes leading/trailing spaces
    from text columns.
    """

    df = df.copy()

    text_cols = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for col in text_cols:

        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
        )

    return df


def remove_exact_duplicates(df):

    """
    Removes completely duplicated rows.
    """

    df = df.copy()

    before = len(df)

    df = df.drop_duplicates()

    removed = before - len(df)

    print(
        f"Exact duplicate rows removed: {removed}"
    )

    return df


def standardize_email(series):

    """
    Standardizes email formatting.
    """

    return (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )


def convert_date_columns(
    df,
    columns
):

    """
    Converts selected columns to datetime.
    Invalid dates become NaT.
    """

    df = df.copy()

    for col in columns:

        if col in df.columns:

            df[col] = pd.to_datetime(
                df[col],
                errors="coerce",
                format="mixed"
            )

    return df


# ============================================================
# CUSTOMERS CLEANING
# ============================================================

def clean_customers(df):

    print("\nCLEANING CUSTOMERS")
    print("-" * 80)

    df = clean_text_columns(df)

    df = remove_exact_duplicates(df)

    # --------------------------------------------------------
    # Remove duplicate customer IDs
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=["customer_id"],
        keep="first"
    )

    print(
        "Duplicate customer IDs removed:",
        before - len(df)
    )

    # --------------------------------------------------------
    # Email
    # --------------------------------------------------------

    if "email" in df.columns:

        df["email"] = standardize_email(
            df["email"]
        )

        email_pattern = (
            r"^[A-Za-z0-9._%+-]+@"
            r"[A-Za-z0-9.-]+\."
            r"[A-Za-z]{2,}$"
        )

        valid_email = df["email"].str.match(
            email_pattern,
            na=False
        )

        df.loc[
            ~valid_email,
            "email"
        ] = pd.NA

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    df = convert_date_columns(
        df,
        [
            "birth_date",
            "signup_date"
        ]
    )

    # --------------------------------------------------------
    # Standardize categories
    # --------------------------------------------------------

    if "country" in df.columns:

        df["country"] = (
            df["country"]
            .str.strip()
            .str.title()
        )

    if "region" in df.columns:

        df["region"] = (
            df["region"]
            .str.strip()
            .str.title()
        )

    if "gender" in df.columns:

        df["gender"] = (
            df["gender"]
            .str.strip()
            .str.title()
        )

    if "customer_segment" in df.columns:

        df["customer_segment"] = (
            df["customer_segment"]
            .str.strip()
        )

    if "acquisition_channel" in df.columns:

        df["acquisition_channel"] = (
            df["acquisition_channel"]
            .str.strip()
        )

    if "loyalty_level" in df.columns:

        df["loyalty_level"] = (
            df["loyalty_level"]
            .str.strip()
            .str.title()
        )
    # --------------------------------------------------------
    # Churn flag
    # --------------------------------------------------------

    if "churn_flag" in df.columns:

        df["churn_flag"] = (
            df["churn_flag"]
            .replace({
                True: 1,
                False: 0
            })
        )

        df["churn_flag"] = pd.to_numeric(
            df["churn_flag"],
            errors="coerce"
        ).astype("Int64")

        df["churn_flag"] = (
            df["churn_flag"]
            .where(
                df["churn_flag"].isin([0, 1]),
                pd.NA
            )
        )

    return df


# ============================================================
# PRODUCTS CLEANING
# ============================================================

def clean_products(df):

    print("\nCLEANING PRODUCTS")
    print("-" * 80)

    df = clean_text_columns(df)

    df = remove_exact_duplicates(df)

    before = len(df)

    df = df.drop_duplicates(
        subset=["product_id"],
        keep="first"
    )

    print(
        "Duplicate product IDs removed:",
        before - len(df)
    )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    for col in [
        "unit_cost",
        "unit_price"
    ]:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            df.loc[
                df[col] <= 0,
                col
            ] = np.nan

    # --------------------------------------------------------
    # Invalid price < cost
    # --------------------------------------------------------

    if (
        "unit_cost" in df.columns
        and
        "unit_price" in df.columns
    ):

        invalid_price = (
            df["unit_price"].notna()
            & df["unit_cost"].notna()
            & (
                df["unit_price"]
                < df["unit_cost"]
            )
        )

        df.loc[
            invalid_price,
            "unit_price"
        ] = np.nan

    # --------------------------------------------------------
    # Launch date
    # --------------------------------------------------------

    df = convert_date_columns(
        df,
        ["launch_date"]
    )

    return df


# ============================================================
# SALES REPRESENTATIVES CLEANING
# ============================================================

def clean_sales_representatives(df):

    print("\nCLEANING SALES REPRESENTATIVES")
    print("-" * 80)

    df = clean_text_columns(df)

    df = remove_exact_duplicates(df)

    before = len(df)

    df = df.drop_duplicates(
        subset=["sales_rep_id"],
        keep="first"
    )

    print(
        "Duplicate sales representative IDs removed:",
        before - len(df)
    )

    if "email" in df.columns:

        df["email"] = standardize_email(
            df["email"]
        )

    df = convert_date_columns(
        df,
        ["hire_date"]
    )

    return df


# ============================================================
# SALES CLEANING
# ============================================================

def clean_sales(
    df,
    customers=None,
    products=None,
    sales_reps=None
):

    print("\nCLEANING SALES")
    print("-" * 80)

    df = clean_text_columns(df)

    df = remove_exact_duplicates(df)

    before = len(df)

    df = df.drop_duplicates(
        subset=["sale_id"],
        keep="first"
    )

    print(
        "Duplicate sale IDs removed:",
        before - len(df)
    )

    # --------------------------------------------------------
    # Convert numeric fields
    # --------------------------------------------------------

    numeric_cols = [
        "quantity",
        "unit_price",
        "discount_pct",
        "gross_amount",
        "discount_amount",
        "net_amount"
    ]

    for col in numeric_cols:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Quantity
    # --------------------------------------------------------

    if "quantity" in df.columns:

        # Zero quantity is invalid.
        df.loc[
            df["quantity"] == 0,
            "quantity"
        ] = np.nan

        # Negative quantity is kept.
        # It may represent a return.

    # --------------------------------------------------------
    # Discount
    # --------------------------------------------------------

    if "discount_pct" in df.columns:

        invalid_discount = (
            (df["discount_pct"] < 0)
            | (df["discount_pct"] > 100)
        )

        df.loc[
            invalid_discount,
            "discount_pct"
        ] = np.nan

    # --------------------------------------------------------
    # Recalculate amounts
    # --------------------------------------------------------

    required = {
        "quantity",
        "unit_price",
        "discount_pct"
    }

    if required.issubset(df.columns):

        df["gross_amount"] = (
            df["quantity"]
            * df["unit_price"]
        )

        df["discount_amount"] = (
            df["gross_amount"]
            * df["discount_pct"]
            / 100
        )

        df["net_amount"] = (
            df["gross_amount"]
            - df["discount_amount"]
        )

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    df = convert_date_columns(
        df,
        ["order_date"]
    )

    # --------------------------------------------------------
    # Foreign keys
    # --------------------------------------------------------

    if customers is not None:

        valid_customers = set(
            customers["customer_id"]
            .dropna()
        )

        invalid_customer = (
            df["customer_id"].notna()
            & ~df["customer_id"].isin(
                valid_customers
            )
        )

        df.loc[
            invalid_customer,
            "customer_id"
        ] = pd.NA

    if products is not None:

        valid_products = set(
            products["product_id"]
            .dropna()
        )

        invalid_product = (
            df["product_id"].notna()
            & ~df["product_id"].isin(
                valid_products
            )
        )

        df.loc[
            invalid_product,
            "product_id"
        ] = pd.NA

    if sales_reps is not None:

        valid_reps = set(
            sales_reps["sales_rep_id"]
            .dropna()
        )

        invalid_rep = (
            df["sales_rep_id"].notna()
            & ~df["sales_rep_id"].isin(
                valid_reps
            )
        )

        df.loc[
            invalid_rep,
            "sales_rep_id"
        ] = pd.NA

    return df


# ============================================================
# LEADS CLEANING
# ============================================================

def clean_leads(df):

    print("\nCLEANING LEADS")
    print("-" * 80)

    df = clean_text_columns(df)

    df = remove_exact_duplicates(df)

    before = len(df)

    df = df.drop_duplicates(
        subset=["lead_id"],
        keep="first"
    )

    print(
        "Duplicate lead IDs removed:",
        before - len(df)
    )

    if "email" in df.columns:

        df["email"] = standardize_email(
            df["email"]
        )

    if "lead_score" in df.columns:

        df["lead_score"] = pd.to_numeric(
        df["lead_score"],
     errors="coerce"
    ).astype("Float64")

        invalid_score = (
            (df["lead_score"] < 0)
            | (df["lead_score"] > 100)
        )

        df.loc[
            invalid_score,
            "lead_score"
        ] = np.nan

    df = convert_date_columns(
        df,
        ["created_date"]
    )

    return df


# ============================================================
# OPPORTUNITIES CLEANING
# ============================================================

def clean_opportunities(
    df,
    leads=None,
    customers=None,
    sales_reps=None
):

    print("\nCLEANING OPPORTUNITIES")
    print("-" * 80)

    df = clean_text_columns(df)

    df = remove_exact_duplicates(df)

    before = len(df)

    df = df.drop_duplicates(
        subset=["opportunity_id"],
        keep="first"
    )

    print(
        "Duplicate opportunity IDs removed:",
        before - len(df)
    )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    if "probability" in df.columns:

        df["probability"] = pd.to_numeric(
            df["probability"],
            errors="coerce"
        ).astype("Float64")

        invalid_probability = (
            (df["probability"] < 0)
            | (df["probability"] > 100)
        )

        df.loc[
            invalid_probability,
            "probability"
        ] = np.nan

    # --------------------------------------------------------
    # Revenue
    # --------------------------------------------------------

    for col in [
        "expected_revenue",
        "actual_revenue"
    ]:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            df.loc[
                df[col] < 0,
                col
            ] = np.nan

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    df = convert_date_columns(
        df,
        [
            "created_date",
            "expected_close_date",
            "actual_close_date"
        ]
    )

    if (
        "created_date" in df.columns
        and
        "expected_close_date"
        in df.columns
    ):

        invalid_date = (
            df["created_date"].notna()
            & df["expected_close_date"].notna()
            & (
                df["expected_close_date"]
                < df["created_date"]
            )
        )

        df.loc[
            invalid_date,
            "expected_close_date"
        ] = pd.NaT

    # --------------------------------------------------------
    # Foreign keys
    # --------------------------------------------------------

    if leads is not None:

        invalid_lead = (
            df["lead_id"].notna()
            & ~df["lead_id"].isin(
                leads["lead_id"]
            )
        )

        df.loc[
            invalid_lead,
            "lead_id"
        ] = pd.NA

    if customers is not None:

        invalid_customer = (
            df["customer_id"].notna()
            & ~df["customer_id"].isin(
                customers["customer_id"]
            )
        )

        df.loc[
            invalid_customer,
            "customer_id"
        ] = pd.NA

    if sales_reps is not None:

        invalid_rep = (
            df["sales_rep_id"].notna()
            & ~df["sales_rep_id"].isin(
                sales_reps["sales_rep_id"]
            )
        )

        df.loc[
            invalid_rep,
            "sales_rep_id"
        ] = pd.NA

    return df


# ============================================================
# CAMPAIGNS CLEANING
# ============================================================

def clean_campaigns(df):

    print("\nCLEANING CAMPAIGNS")
    print("-" * 80)

    df = clean_text_columns(df)

    df = remove_exact_duplicates(df)

    before = len(df)

    df = df.drop_duplicates(
        subset=["campaign_id"],
        keep="first"
    )

    print(
        "Duplicate campaign IDs removed:",
        before - len(df)
    )

    if "budget" in df.columns:

        df["budget"] = pd.to_numeric(
            df["budget"],
            errors="coerce"
        )

        df.loc[
            df["budget"] < 0,
            "budget"
        ] = np.nan

    df = convert_date_columns(
        df,
        [
            "start_date",
            "end_date"
        ]
    )

    if (
        "start_date" in df.columns
        and
        "end_date" in df.columns
    ):

        invalid_date = (
            df["start_date"].notna()
            & df["end_date"].notna()
            & (
                df["end_date"]
                < df["start_date"]
            )
        )

        df.loc[
            invalid_date,
            "end_date"
        ] = pd.NaT

    return df


# ============================================================
# CAMPAIGN INTERACTIONS CLEANING
# ============================================================

def clean_campaign_interactions(
    df,
    campaigns=None,
    customers=None
):

    print("\nCLEANING CAMPAIGN INTERACTIONS")
    print("-" * 80)

    df = clean_text_columns(df)

    df = remove_exact_duplicates(df)

    before = len(df)

    df = df.drop_duplicates(
        subset=["interaction_id"],
        keep="first"
    )

    print(
        "Duplicate interaction IDs removed:",
        before - len(df)
    )

    df = convert_date_columns(
        df,
        ["interaction_date"]
    )

        # --------------------------------------------------------
    # Boolean / binary columns
    # --------------------------------------------------------

    for col in [
        "opened",
        "clicked",
        "converted"
    ]:

        if col in df.columns:

            df[col] = (
                df[col]
                .replace({
                    True: 1,
                    False: 0
                })
            )

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).astype("Int64")

            df[col] = df[col].where(
                df[col].isin([0, 1]),
                pd.NA
            )

    # --------------------------------------------------------
    # Conversion value
    # --------------------------------------------------------

    if "conversion_value" in df.columns:

        df["conversion_value"] = pd.to_numeric(
            df["conversion_value"],
            errors="coerce"
        )

        df.loc[
            df["conversion_value"] < 0,
            "conversion_value"
        ] = np.nan

    # --------------------------------------------------------
    # Foreign keys
    # --------------------------------------------------------

    if campaigns is not None:

        invalid_campaign = (
            df["campaign_id"].notna()
            & ~df["campaign_id"].isin(
                campaigns["campaign_id"]
            )
        )

        df.loc[
            invalid_campaign,
            "campaign_id"
        ] = pd.NA

    if customers is not None:

        invalid_customer = (
            df["customer_id"].notna()
            & ~df["customer_id"].isin(
                customers["customer_id"]
            )
        )

        df.loc[
            invalid_customer,
            "customer_id"
        ] = pd.NA

    return df


# ============================================================
# SUPPORT TICKETS CLEANING
# ============================================================

def clean_support_tickets(
    df,
    customers=None
):

    print("\nCLEANING SUPPORT TICKETS")
    print("-" * 80)

    df = clean_text_columns(df)

    df = remove_exact_duplicates(df)

    before = len(df)

    df = df.drop_duplicates(
        subset=["ticket_id"],
        keep="first"
    )

    print(
        "Duplicate ticket IDs removed:",
        before - len(df)
    )

    # --------------------------------------------------------
    # Foreign key
    # --------------------------------------------------------

    if customers is not None:

        invalid_customer = (
            df["customer_id"].notna()
            & ~df["customer_id"].isin(
                customers["customer_id"]
            )
        )

        df.loc[
            invalid_customer,
            "customer_id"
        ] = pd.NA

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    df = convert_date_columns(
        df,
        [
            "created_date",
            "resolved_date"
        ]
    )

    if (
        "created_date" in df.columns
        and
        "resolved_date" in df.columns
    ):

        invalid_date = (
            df["created_date"].notna()
            & df["resolved_date"].notna()
            & (
                df["resolved_date"]
                < df["created_date"]
            )
        )

        df.loc[
            invalid_date,
            "resolved_date"
        ] = pd.NaT

    # --------------------------------------------------------
    # Resolution hours
    # --------------------------------------------------------

    if "resolution_hours" in df.columns:

        df["resolution_hours"] = pd.to_numeric(
            df["resolution_hours"],
            errors="coerce"
        )

        df.loc[
            df["resolution_hours"] < 0,
            "resolution_hours"
        ] = np.nan

        # Values > 720 are kept.
        # They are outliers but may represent real long cases.

    # --------------------------------------------------------
    # Satisfaction
    # --------------------------------------------------------

    if "satisfaction_score" in df.columns:

        df["satisfaction_score"] = pd.to_numeric(
            df["satisfaction_score"],
            errors="coerce"
        )

        invalid_satisfaction = (
            (df["satisfaction_score"] < 1)
            | (df["satisfaction_score"] > 5)
        )

        df.loc[
            invalid_satisfaction,
            "satisfaction_score"
        ] = np.nan

    return df


# ============================================================
# CLEAN TABLES
# ============================================================

cleaned = {}


if "customers" in dataframes:

    cleaned["customers"] = clean_customers(
        dataframes["customers"]
    )


if "products" in dataframes:

    cleaned["products"] = clean_products(
        dataframes["products"]
    )


if "sales_representatives" in dataframes:

    cleaned[
        "sales_representatives"
    ] = clean_sales_representatives(
        dataframes["sales_representatives"]
    )


if "leads" in dataframes:

    cleaned["leads"] = clean_leads(
        dataframes["leads"]
    )


if "campaigns" in dataframes:

    cleaned["campaigns"] = clean_campaigns(
        dataframes["campaigns"]
    )


# ============================================================
# CLEAN TABLES THAT DEPEND ON OTHER TABLES
# ============================================================

if "sales" in dataframes:

    cleaned["sales"] = clean_sales(
        dataframes["sales"],
        cleaned.get("customers"),
        cleaned.get("products"),
        cleaned.get(
            "sales_representatives"
        )
    )


if "opportunities" in dataframes:

    cleaned[
        "opportunities"
    ] = clean_opportunities(
        dataframes["opportunities"],
        cleaned.get("leads"),
        cleaned.get("customers"),
        cleaned.get(
            "sales_representatives"
        )
    )


if "campaign_interactions" in dataframes:

    cleaned[
        "campaign_interactions"
    ] = clean_campaign_interactions(
        dataframes[
            "campaign_interactions"
        ],
        cleaned.get("campaigns"),
        cleaned.get("customers")
    )


if "support_tickets" in dataframes:

    cleaned[
        "support_tickets"
    ] = clean_support_tickets(
        dataframes[
            "support_tickets"
        ],
        cleaned.get("customers")
    )


# ============================================================
# SAVE CLEANED DATA
# ============================================================

print("\n" + "=" * 80)
print("SAVING CLEANED DATA")
print("=" * 80)


for table_name, df in cleaned.items():

    output_path = (
        PROCESSED_DIR /
        f"{table_name}.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"[SAVED] {table_name}: "
        f"{len(df):,} rows "
        f"→ {output_path}"
    )


print(
    "\nDATA CLEANING COMPLETED SUCCESSFULLY."
)