import pandas as pd
import numpy as np
import re
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROFILING_DIR = PROJECT_ROOT / "data" / "profiling"

PROFILING_DIR.mkdir(parents=True, exist_ok=True)

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
# LOAD DATA
# ============================================================

dataframes = {}

for table_name, filename in FILES.items():
    path = RAW_DIR / filename

    if path.exists():
        dataframes[table_name] = pd.read_csv(path)

        print(
            f"[OK] {table_name}: "
            f"{len(dataframes[table_name]):,} rows"
        )
    else:
        print(f"[MISSING] {path}")


# ============================================================
# GENERIC PROFILING
# ============================================================

def profile_table(df, table_name):

    print("\n" + "=" * 80)
    print(f"PROFILING TABLE: {table_name.upper()}")
    print("=" * 80)

    print(f"\nRows       : {len(df):,}")
    print(f"Columns    : {len(df.columns)}")
    print(f"Duplicates : {df.duplicated().sum():,}")

    print("\nCOLUMN TYPES")
    print("-" * 80)
    print(df.dtypes)

    print("\nMISSING VALUES")
    print("-" * 80)

    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": (
            df.isna().mean() * 100
        ).round(2)
    })

    print(
        missing.sort_values(
            "missing_count",
            ascending=False
        )
    )

    print("\nUNIQUE VALUES")
    print("-" * 80)

    unique = pd.DataFrame({
        "unique_values":
        df.nunique(dropna=False)
    })

    print(unique)

    print("\nNUMERIC STATISTICS")
    print("-" * 80)

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe().T)

    # --------------------------------------------------------
    # CREATE PROFILE FILE
    # --------------------------------------------------------

    profile = []

    for col in df.columns:

        profile.append({
            "table": table_name,
            "column": col,
            "dtype": str(df[col].dtype),
            "rows": len(df),
            "missing_count":
                int(df[col].isna().sum()),
            "missing_pct":
                round(
                    df[col].isna().mean() * 100,
                    2
                ),
            "unique_values":
                int(
                    df[col].nunique(
                        dropna=False
                    )
                )
        })

    profile_df = pd.DataFrame(profile)

    output_path = (
        PROFILING_DIR /
        f"{table_name}_profile.csv"
    )

    profile_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nProfile saved: {output_path}"
    )


# ============================================================
# GENERIC TEXT QUALITY CHECK
# ============================================================

def check_text_quality(df, table_name):

    print(
        f"\nTEXT QUALITY CHECK - {table_name}"
    )
    print("-" * 80)

    object_cols = df.select_dtypes(
        include=["object"]
    ).columns

    for col in object_cols:

        series = df[col].astype("string")

        whitespace_mask = (
            series.notna()
            & (series != series.str.strip())
        )

        whitespace_count = (
            whitespace_mask.sum()
        )

        if whitespace_count > 0:
            print(
                f"{col}: "
                f"{whitespace_count:,} values "
                "with leading/trailing spaces"
            )


# ============================================================
# CUSTOMERS CHECKS
# ============================================================

def check_customers(df):

    print("\nCUSTOMERS BUSINESS CHECKS")
    print("-" * 80)

    # Duplicate IDs
    if "customer_id" in df.columns:

        duplicated = (
            df["customer_id"]
            .duplicated(keep=False)
        )

        print(
            "Duplicated customer_id:",
            duplicated.sum()
        )

    # Email validation
    if "email" in df.columns:

        email_pattern = (
            r"^[A-Za-z0-9._%+-]+@"
            r"[A-Za-z0-9.-]+\."
            r"[A-Za-z]{2,}$"
        )

        emails = df["email"].astype(
            "string"
        )

        invalid = (
            emails.notna()
            & ~emails.str.match(
                email_pattern,
                na=False
            )
        )

        print(
            "Invalid emails:",
            invalid.sum()
        )

    # Date validation
    for col in [
        "birth_date",
        "signup_date"
    ]:

        if col in df.columns:

            parsed = pd.to_datetime(
                df[col],
                errors="coerce",
                format="mixed"
            )

            invalid = (
                df[col].notna()
                & parsed.isna()
            )

            print(
                f"Invalid {col}:",
                invalid.sum()
            )

    # Category distributions
    for col in [
        "customer_segment",
        "acquisition_channel",
        "loyalty_level",
        "country",
        "region",
        "gender",
        "churn_flag"
    ]:

        if col in df.columns:

            print(
                f"\n{col}:"
            )

            print(
                df[col]
                .astype("string")
                .value_counts(
                    dropna=False
                )
                .head(30)
            )


# ============================================================
# PRODUCTS CHECKS
# ============================================================

def check_products(df):

    print("\nPRODUCTS BUSINESS CHECKS")
    print("-" * 80)

    if "product_id" in df.columns:
        print(
            "Duplicated product_id:",
            df["product_id"]
            .duplicated()
            .sum()
        )

    for col in [
        "unit_cost",
        "unit_price"
    ]:

        if col in df.columns:

            print(
                f"{col} <= 0:",
                (
                    pd.to_numeric(
                        df[col],
                        errors="coerce"
                    ) <= 0
                ).sum()
            )

    if (
        "unit_cost" in df.columns
        and "unit_price" in df.columns
    ):

        cost = pd.to_numeric(
            df["unit_cost"],
            errors="coerce"
        )

        price = pd.to_numeric(
            df["unit_price"],
            errors="coerce"
        )

        print(
            "Products with price < cost:",
            (price < cost).sum()
        )


# ============================================================
# SALES REPRESENTATIVES CHECKS
# ============================================================

def check_sales_representatives(df):

    print("\nSALES REPRESENTATIVES CHECKS")
    print("-" * 80)

    if "sales_rep_id" in df.columns:

        print(
            "Duplicated sales_rep_id:",
            df["sales_rep_id"]
            .duplicated()
            .sum()
        )

    if "email" in df.columns:

        print(
            "Missing emails:",
            df["email"]
            .isna()
            .sum()
        )


# ============================================================
# SALES CHECKS
# ============================================================

def check_sales(
    df,
    customers=None,
    products=None,
    sales_reps=None
):

    print("\nSALES BUSINESS CHECKS")
    print("-" * 80)

    if "quantity" in df.columns:

        quantity = pd.to_numeric(
            df["quantity"],
            errors="coerce"
        )

        print(
            "Negative quantities:",
            (quantity < 0).sum()
        )

        print(
            "Zero quantities:",
            (quantity == 0).sum()
        )

    if "discount_pct" in df.columns:

        discount = pd.to_numeric(
            df["discount_pct"],
            errors="coerce"
        )

        print(
            "Discount < 0:",
            (discount < 0).sum()
        )

        print(
            "Discount > 100:",
            (discount > 100).sum()
        )

    # Amount consistency
    required = {
        "quantity",
        "unit_price",
        "discount_pct",
        "gross_amount",
        "discount_amount",
        "net_amount"
    }

    if required.issubset(df.columns):

        quantity = pd.to_numeric(
            df["quantity"],
            errors="coerce"
        )

        price = pd.to_numeric(
            df["unit_price"],
            errors="coerce"
        )

        discount_pct = pd.to_numeric(
            df["discount_pct"],
            errors="coerce"
        )

        gross = pd.to_numeric(
            df["gross_amount"],
            errors="coerce"
        )

        discount_amount = pd.to_numeric(
            df["discount_amount"],
            errors="coerce"
        )

        net = pd.to_numeric(
            df["net_amount"],
            errors="coerce"
        )

        expected_gross = (
            quantity * price
        )

        expected_discount = (
            expected_gross
            * discount_pct
            / 100
        )

        expected_net = (
            expected_gross
            - expected_discount
        )

        gross_error = (
            ~np.isclose(
                gross,
                expected_gross,
                equal_nan=True
            )
        )

        net_error = (
            ~np.isclose(
                net,
                expected_net,
                equal_nan=True
            )
        )

        print(
            "Gross amount inconsistencies:",
            gross_error.sum()
        )

        print(
            "Net amount inconsistencies:",
            net_error.sum()
        )

    # Foreign keys
    if (
        customers is not None
        and "customer_id" in df.columns
    ):

        invalid = (
            ~df["customer_id"].isin(
                customers["customer_id"]
            )
        )

        print(
            "Invalid customer_id:",
            invalid.sum()
        )

    if (
        products is not None
        and "product_id" in df.columns
    ):

        invalid = (
            ~df["product_id"].isin(
                products["product_id"]
            )
        )

        print(
            "Invalid product_id:",
            invalid.sum()
        )

    if (
        sales_reps is not None
        and "sales_rep_id" in df.columns
    ):

        invalid = (
            ~df["sales_rep_id"].isin(
                sales_reps["sales_rep_id"]
            )
        )

        print(
            "Invalid sales_rep_id:",
            invalid.sum()
        )


# ============================================================
# LEADS CHECKS
# ============================================================

def check_leads(df):

    print("\nLEADS BUSINESS CHECKS")
    print("-" * 80)

    if "lead_score" in df.columns:

        score = pd.to_numeric(
            df["lead_score"],
            errors="coerce"
        )

        print(
            "Lead score < 0:",
            (score < 0).sum()
        )

        print(
            "Lead score > 100:",
            (score > 100).sum()
        )

    if "lead_status" in df.columns:

        print(
            "\nLead statuses:"
        )

        print(
            df["lead_status"]
            .astype("string")
            .value_counts(
                dropna=False
            )
        )


# ============================================================
# OPPORTUNITIES CHECKS
# ============================================================

def check_opportunities(
    df,
    leads=None,
    customers=None,
    sales_reps=None
):

    print("\nOPPORTUNITIES BUSINESS CHECKS")
    print("-" * 80)

    if "probability" in df.columns:

        probability = pd.to_numeric(
            df["probability"],
            errors="coerce"
        )

        print(
            "Probability < 0:",
            (probability < 0).sum()
        )

        print(
            "Probability > 100:",
            (probability > 100).sum()
        )

    if (
        "created_date" in df.columns
        and
        "expected_close_date"
        in df.columns
    ):

        created = pd.to_datetime(
            df["created_date"],
            errors="coerce",
            format="mixed"
        )

        expected_close = pd.to_datetime(
            df["expected_close_date"],
            errors="coerce",
            format="mixed"
        )

        invalid = (
            expected_close < created
        )

        print(
            "Expected close before creation:",
            invalid.sum()
        )

    if leads is not None:

        invalid = (
            ~df["lead_id"].isin(
                leads["lead_id"]
            )
        )

        print(
            "Invalid lead_id:",
            invalid.sum()
        )

    if customers is not None:

        mask = (
            df["customer_id"]
            .notna()
        )

        invalid = (
            mask
            & ~df["customer_id"]
            .isin(
                customers[
                    "customer_id"
                ]
            )
        )

        print(
            "Invalid customer_id:",
            invalid.sum()
        )

    if sales_reps is not None:

        invalid = (
            ~df["sales_rep_id"]
            .isin(
                sales_reps[
                    "sales_rep_id"
                ]
            )
        )

        print(
            "Invalid sales_rep_id:",
            invalid.sum()
        )


# ============================================================
# CAMPAIGNS CHECKS
# ============================================================

def check_campaigns(df):

    print("\nCAMPAIGNS BUSINESS CHECKS")
    print("-" * 80)

    if "budget" in df.columns:

        budget = pd.to_numeric(
            df["budget"],
            errors="coerce"
        )

        print(
            "Negative budget:",
            (budget < 0).sum()
        )

    if (
        "start_date" in df.columns
        and
        "end_date" in df.columns
    ):

        start = pd.to_datetime(
            df["start_date"],
            errors="coerce",
            format="mixed"
        )

        end = pd.to_datetime(
            df["end_date"],
            errors="coerce",
            format="mixed"
        )

        print(
            "End date before start date:",
            (end < start).sum()
        )


# ============================================================
# CAMPAIGN INTERACTIONS CHECKS
# ============================================================

def check_campaign_interactions(
    df,
    campaigns=None,
    customers=None
):

    print("\nCAMPAIGN INTERACTIONS CHECKS")
    print("-" * 80)

    if campaigns is not None:

        invalid = (
            ~df["campaign_id"].isin(
                campaigns[
                    "campaign_id"
                ]
            )
        )

        print(
            "Invalid campaign_id:",
            invalid.sum()
        )

    if customers is not None:

        invalid = (
            ~df["customer_id"].isin(
                customers[
                    "customer_id"
                ]
            )
        )

        print(
            "Invalid customer_id:",
            invalid.sum()
        )

    if "conversion_value" in df.columns:

        values = pd.to_numeric(
            df["conversion_value"],
            errors="coerce"
        )

        print(
            "Negative conversion value:",
            (values < 0).sum()
        )


# ============================================================
# SUPPORT TICKETS CHECKS
# ============================================================

def check_support_tickets(
    df,
    customers=None
):

    print("\nSUPPORT TICKETS BUSINESS CHECKS")
    print("-" * 80)

    if customers is not None:

        invalid = (
            ~df["customer_id"].isin(
                customers[
                    "customer_id"
                ]
            )
        )

        print(
            "Invalid customer_id:",
            invalid.sum()
        )

    if (
        "created_date" in df.columns
        and
        "resolved_date" in df.columns
    ):

        created = pd.to_datetime(
            df["created_date"],
            errors="coerce",
            format="mixed"
        )

        resolved = pd.to_datetime(
            df["resolved_date"],
            errors="coerce",
            format="mixed"
        )

        invalid = (
            resolved.notna()
            & created.notna()
            & (resolved < created)
        )

        print(
            "Resolved before created:",
            invalid.sum()
        )

    if "resolution_hours" in df.columns:

        hours = pd.to_numeric(
            df["resolution_hours"],
            errors="coerce"
        )

        print(
            "Negative resolution_hours:",
            (hours < 0).sum()
        )

        print(
            "Very high resolution_hours (>720):",
            (hours > 720).sum()
        )

    if "satisfaction_score" in df.columns:

        satisfaction = pd.to_numeric(
            df["satisfaction_score"],
            errors="coerce"
        )

        invalid = (
            satisfaction.notna()
            & (
                (satisfaction < 1)
                | (satisfaction > 5)
            )
        )

        print(
            "Invalid satisfaction scores:",
            invalid.sum()
        )


# ============================================================
# RUN GENERAL PROFILING
# ============================================================

for table_name, df in dataframes.items():

    profile_table(
        df,
        table_name
    )

    check_text_quality(
        df,
        table_name
    )


# ============================================================
# RUN BUSINESS-SPECIFIC CHECKS
# ============================================================

if "customers" in dataframes:
    check_customers(
        dataframes["customers"]
    )

if "products" in dataframes:
    check_products(
        dataframes["products"]
    )

if "sales_representatives" in dataframes:
    check_sales_representatives(
        dataframes[
            "sales_representatives"
        ]
    )

if "sales" in dataframes:

    check_sales(
        dataframes["sales"],
        dataframes.get("customers"),
        dataframes.get("products"),
        dataframes.get(
            "sales_representatives"
        )
    )

if "leads" in dataframes:
    check_leads(
        dataframes["leads"]
    )

if "opportunities" in dataframes:

    check_opportunities(
        dataframes["opportunities"],
        dataframes.get("leads"),
        dataframes.get("customers"),
        dataframes.get(
            "sales_representatives"
        )
    )

if "campaigns" in dataframes:
    check_campaigns(
        dataframes["campaigns"]
    )

if "campaign_interactions" in dataframes:

    check_campaign_interactions(
        dataframes[
            "campaign_interactions"
        ],
        dataframes.get("campaigns"),
        dataframes.get("customers")
    )

if "support_tickets" in dataframes:

    check_support_tickets(
        dataframes[
            "support_tickets"
        ],
        dataframes.get("customers")
    )


# ============================================================
# GLOBAL DATA QUALITY SUMMARY
# ============================================================

summary = []

for table_name, df in dataframes.items():

    summary.append({
        "table": table_name,
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values":
            int(
                df.isna()
                .sum()
                .sum()
            ),
        "duplicate_rows":
            int(
                df.duplicated()
                .sum()
            )
    })

summary_df = pd.DataFrame(summary)

summary_path = (
    PROFILING_DIR /
    "global_data_quality_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

print("\n" + "=" * 80)
print("GLOBAL DATA QUALITY SUMMARY")
print("=" * 80)

print(summary_df)

print(
    f"\nGlobal summary saved to:\n"
    f"{summary_path}"
)

print("\nDATA PROFILING COMPLETED SUCCESSFULLY.")