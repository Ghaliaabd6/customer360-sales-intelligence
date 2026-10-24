import pandas as pd
from pathlib import Path
import re

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
QUALITY_DIR = PROJECT_ROOT / "data" / "quality"

QUALITY_DIR.mkdir(parents=True, exist_ok=True)

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

        dataframes[table_name] = pd.read_csv(
            path,
            low_memory=False
        )

        print(
            f"[OK] {table_name}: "
            f"{len(dataframes[table_name]):,} rows"
        )

    else:
        print(f"[MISSING] {path}")


# ============================================================
# STORE QUALITY RESULTS
# ============================================================

quality_results = []


def add_result(
    table,
    column,
    rule,
    issue_count,
    severity="Medium"
):

    quality_results.append({
        "table": table,
        "column": column,
        "rule": rule,
        "issue_count": int(issue_count),
        "severity": severity
    })


# ============================================================
# CUSTOMERS RULES
# ============================================================

def validate_customers(df):

    print("\nCUSTOMERS QUALITY RULES")
    print("-" * 80)

    # --------------------------------------------------------
    # Rule 1 - customer_id must not be null
    # --------------------------------------------------------

    missing_id = df["customer_id"].isna().sum()

    add_result(
        "customers",
        "customer_id",
        "Customer ID must not be null",
        missing_id,
        "Critical"
    )

    # --------------------------------------------------------
    # Rule 2 - customer_id must be unique
    # --------------------------------------------------------

    duplicate_id = (
        df["customer_id"]
        .duplicated(keep=False)
        .sum()
    )

    add_result(
        "customers",
        "customer_id",
        "Customer ID must be unique",
        duplicate_id,
        "Critical"
    )

    # --------------------------------------------------------
    # Rule 3 - email format
    # --------------------------------------------------------

    email_pattern = (
        r"^[A-Za-z0-9._%+-]+@"
        r"[A-Za-z0-9.-]+\."
        r"[A-Za-z]{2,}$"
    )

    emails = df["email"].astype("string")

    invalid_email = (
        emails.notna()
        & ~emails.str.match(
            email_pattern,
            na=False
        )
    ).sum()

    add_result(
        "customers",
        "email",
        "Email must have a valid format",
        invalid_email,
        "High"
    )

    # --------------------------------------------------------
    # Rule 4 - churn_flag must be 0 or 1
    # --------------------------------------------------------

    if "churn_flag" in df.columns:

        churn = pd.to_numeric(
            df["churn_flag"],
            errors="coerce"
        )

        invalid_churn = (
            churn.notna()
            & ~churn.isin([0, 1])
        ).sum()

        add_result(
            "customers",
            "churn_flag",
            "Churn flag must be 0 or 1",
            invalid_churn,
            "High"
        )

    # --------------------------------------------------------
    # Rule 5 - signup_date must be valid
    # --------------------------------------------------------

    if "signup_date" in df.columns:

        signup = pd.to_datetime(
            df["signup_date"],
            errors="coerce",
            format="mixed"
        )

        invalid_signup = (
            df["signup_date"].notna()
            & signup.isna()
        ).sum()

        add_result(
            "customers",
            "signup_date",
            "Signup date must be a valid date",
            invalid_signup,
            "High"
        )


# ============================================================
# PRODUCTS RULES
# ============================================================

def validate_products(df):

    print("\nPRODUCTS QUALITY RULES")
    print("-" * 80)

    missing_id = df["product_id"].isna().sum()

    add_result(
        "products",
        "product_id",
        "Product ID must not be null",
        missing_id,
        "Critical"
    )

    duplicate_id = (
        df["product_id"]
        .duplicated(keep=False)
        .sum()
    )

    add_result(
        "products",
        "product_id",
        "Product ID must be unique",
        duplicate_id,
        "Critical"
    )

    unit_cost = pd.to_numeric(
        df["unit_cost"],
        errors="coerce"
    )

    unit_price = pd.to_numeric(
        df["unit_price"],
        errors="coerce"
    )

    invalid_cost = (
        unit_cost.notna()
        & (unit_cost <= 0)
    ).sum()

    add_result(
        "products",
        "unit_cost",
        "Unit cost must be greater than 0",
        invalid_cost,
        "High"
    )

    invalid_price = (
        unit_price.notna()
        & (unit_price <= 0)
    ).sum()

    add_result(
        "products",
        "unit_price",
        "Unit price must be greater than 0",
        invalid_price,
        "High"
    )

    below_cost = (
        unit_price.notna()
        & unit_cost.notna()
        & (unit_price < unit_cost)
    ).sum()

    add_result(
        "products",
        "unit_price",
        "Unit price should not be lower than unit cost",
        below_cost,
        "Medium"
    )


# ============================================================
# SALES REPRESENTATIVES RULES
# ============================================================

def validate_sales_representatives(df):

    print("\nSALES REPRESENTATIVES QUALITY RULES")
    print("-" * 80)

    missing_id = df["sales_rep_id"].isna().sum()

    add_result(
        "sales_representatives",
        "sales_rep_id",
        "Sales representative ID must not be null",
        missing_id,
        "Critical"
    )

    duplicate_id = (
        df["sales_rep_id"]
        .duplicated(keep=False)
        .sum()
    )

    add_result(
        "sales_representatives",
        "sales_rep_id",
        "Sales representative ID must be unique",
        duplicate_id,
        "Critical"
    )

    if "email" in df.columns:

        missing_email = df["email"].isna().sum()

        add_result(
            "sales_representatives",
            "email",
            "Sales representative email should not be null",
            missing_email,
            "Medium"
        )


# ============================================================
# SALES RULES
# ============================================================

def validate_sales(
    df,
    customers=None,
    products=None,
    sales_reps=None
):

    print("\nSALES QUALITY RULES")
    print("-" * 80)

    missing_sale_id = df["sale_id"].isna().sum()

    add_result(
        "sales",
        "sale_id",
        "Sale ID must not be null",
        missing_sale_id,
        "Critical"
    )

    duplicate_sale_id = (
        df["sale_id"]
        .duplicated(keep=False)
        .sum()
    )

    add_result(
        "sales",
        "sale_id",
        "Sale ID must be unique",
        duplicate_sale_id,
        "Critical"
    )

    quantity = pd.to_numeric(
        df["quantity"],
        errors="coerce"
    )

    zero_quantity = (
        quantity.notna()
        & (quantity == 0)
    ).sum()

    add_result(
        "sales",
        "quantity",
        "Quantity must not be zero",
        zero_quantity,
        "High"
    )

    # Negative quantity may represent returns.
    negative_quantity = (
        quantity.notna()
        & (quantity < 0)
    ).sum()

    add_result(
        "sales",
        "quantity",
        "Negative quantity should be reviewed as a possible return",
        negative_quantity,
        "Medium"
    )

    discount = pd.to_numeric(
        df["discount_pct"],
        errors="coerce"
    )

    invalid_discount = (
        discount.notna()
        & (
            (discount < 0)
            | (discount > 100)
        )
    ).sum()

    add_result(
        "sales",
        "discount_pct",
        "Discount percentage must be between 0 and 100",
        invalid_discount,
        "High"
    )

    # --------------------------------------------------------
    # Referential integrity
    # --------------------------------------------------------

    if customers is not None:

        invalid_customer = (
            df["customer_id"].notna()
            & ~df["customer_id"].isin(
                customers["customer_id"]
            )
        ).sum()

        add_result(
            "sales",
            "customer_id",
            "Customer ID must exist in customers",
            invalid_customer,
            "Critical"
        )

    if products is not None:

        invalid_product = (
            df["product_id"].notna()
            & ~df["product_id"].isin(
                products["product_id"]
            )
        ).sum()

        add_result(
            "sales",
            "product_id",
            "Product ID must exist in products",
            invalid_product,
            "Critical"
        )

    if sales_reps is not None:

        invalid_rep = (
            df["sales_rep_id"].notna()
            & ~df["sales_rep_id"].isin(
                sales_reps["sales_rep_id"]
            )
        ).sum()

        add_result(
            "sales",
            "sales_rep_id",
            "Sales representative ID must exist",
            invalid_rep,
            "Critical"
        )


# ============================================================
# LEADS RULES
# ============================================================

def validate_leads(df):

    print("\nLEADS QUALITY RULES")
    print("-" * 80)

    missing_id = df["lead_id"].isna().sum()

    add_result(
        "leads",
        "lead_id",
        "Lead ID must not be null",
        missing_id,
        "Critical"
    )

    duplicate_id = (
        df["lead_id"]
        .duplicated(keep=False)
        .sum()
    )

    add_result(
        "leads",
        "lead_id",
        "Lead ID must be unique",
        duplicate_id,
        "Critical"
    )

    score = pd.to_numeric(
        df["lead_score"],
        errors="coerce"
    )

    invalid_score = (
        score.notna()
        & (
            (score < 0)
            | (score > 100)
        )
    ).sum()

    add_result(
        "leads",
        "lead_score",
        "Lead score must be between 0 and 100",
        invalid_score,
        "High"
    )


# ============================================================
# OPPORTUNITIES RULES
# ============================================================

def validate_opportunities(
    df,
    leads=None,
    customers=None,
    sales_reps=None
):

    print("\nOPPORTUNITIES QUALITY RULES")
    print("-" * 80)

    missing_id = df["opportunity_id"].isna().sum()

    add_result(
        "opportunities",
        "opportunity_id",
        "Opportunity ID must not be null",
        missing_id,
        "Critical"
    )

    duplicate_id = (
        df["opportunity_id"]
        .duplicated(keep=False)
        .sum()
    )

    add_result(
        "opportunities",
        "opportunity_id",
        "Opportunity ID must be unique",
        duplicate_id,
        "Critical"
    )

    probability = pd.to_numeric(
        df["probability"],
        errors="coerce"
    )

    invalid_probability = (
        probability.notna()
        & (
            (probability < 0)
            | (probability > 100)
        )
    ).sum()

    add_result(
        "opportunities",
        "probability",
        "Probability must be between 0 and 100",
        invalid_probability,
        "High"
    )

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

    invalid_dates = (
        created.notna()
        & expected_close.notna()
        & (expected_close < created)
    ).sum()

    add_result(
        "opportunities",
        "expected_close_date",
        "Expected close date must not be before creation date",
        invalid_dates,
        "High"
    )

    if leads is not None:

        invalid_lead = (
            df["lead_id"].notna()
            & ~df["lead_id"].isin(
                leads["lead_id"]
            )
        ).sum()

        add_result(
            "opportunities",
            "lead_id",
            "Lead ID must exist in leads",
            invalid_lead,
            "Critical"
        )

    if customers is not None:

        invalid_customer = (
            df["customer_id"].notna()
            & ~df["customer_id"].isin(
                customers["customer_id"]
            )
        ).sum()

        add_result(
            "opportunities",
            "customer_id",
            "Customer ID must exist when present",
            invalid_customer,
            "High"
        )

    if sales_reps is not None:

        invalid_rep = (
            df["sales_rep_id"].notna()
            & ~df["sales_rep_id"].isin(
                sales_reps["sales_rep_id"]
            )
        ).sum()

        add_result(
            "opportunities",
            "sales_rep_id",
            "Sales representative ID must exist",
            invalid_rep,
            "Critical"
        )


# ============================================================
# CAMPAIGNS RULES
# ============================================================

def validate_campaigns(df):

    print("\nCAMPAIGNS QUALITY RULES")
    print("-" * 80)

    missing_id = df["campaign_id"].isna().sum()

    add_result(
        "campaigns",
        "campaign_id",
        "Campaign ID must not be null",
        missing_id,
        "Critical"
    )

    duplicate_id = (
        df["campaign_id"]
        .duplicated(keep=False)
        .sum()
    )

    add_result(
        "campaigns",
        "campaign_id",
        "Campaign ID must be unique",
        duplicate_id,
        "Critical"
    )

    budget = pd.to_numeric(
        df["budget"],
        errors="coerce"
    )

    negative_budget = (
        budget.notna()
        & (budget < 0)
    ).sum()

    add_result(
        "campaigns",
        "budget",
        "Campaign budget must not be negative",
        negative_budget,
        "High"
    )

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

    invalid_dates = (
        start.notna()
        & end.notna()
        & (end < start)
    ).sum()

    add_result(
        "campaigns",
        "end_date",
        "Campaign end date must not be before start date",
        invalid_dates,
        "High"
    )


# ============================================================
# CAMPAIGN INTERACTIONS RULES
# ============================================================

def validate_campaign_interactions(
    df,
    campaigns=None,
    customers=None
):

    print("\nCAMPAIGN INTERACTIONS QUALITY RULES")
    print("-" * 80)

    missing_id = df["interaction_id"].isna().sum()

    add_result(
        "campaign_interactions",
        "interaction_id",
        "Interaction ID must not be null",
        missing_id,
        "Critical"
    )

    if campaigns is not None:

        invalid_campaign = (
            df["campaign_id"].notna()
            & ~df["campaign_id"].isin(
                campaigns["campaign_id"]
            )
        ).sum()

        add_result(
            "campaign_interactions",
            "campaign_id",
            "Campaign ID must exist in campaigns",
            invalid_campaign,
            "Critical"
        )

    if customers is not None:

        invalid_customer = (
            df["customer_id"].notna()
            & ~df["customer_id"].isin(
                customers["customer_id"]
            )
        ).sum()

        add_result(
            "campaign_interactions",
            "customer_id",
            "Customer ID must exist in customers",
            invalid_customer,
            "Critical"
        )

    conversion_value = pd.to_numeric(
        df["conversion_value"],
        errors="coerce"
    )

    negative_conversion = (
        conversion_value.notna()
        & (conversion_value < 0)
    ).sum()

    add_result(
        "campaign_interactions",
        "conversion_value",
        "Conversion value must not be negative",
        negative_conversion,
        "High"
    )


# ============================================================
# SUPPORT TICKETS RULES
# ============================================================

def validate_support_tickets(
    df,
    customers=None
):

    print("\nSUPPORT TICKETS QUALITY RULES")
    print("-" * 80)

    missing_id = df["ticket_id"].isna().sum()

    add_result(
        "support_tickets",
        "ticket_id",
        "Ticket ID must not be null",
        missing_id,
        "Critical"
    )

    duplicate_id = (
        df["ticket_id"]
        .duplicated(keep=False)
        .sum()
    )

    add_result(
        "support_tickets",
        "ticket_id",
        "Ticket ID must be unique",
        duplicate_id,
        "Critical"
    )

    if customers is not None:

        invalid_customer = (
            df["customer_id"].notna()
            & ~df["customer_id"].isin(
                customers["customer_id"]
            )
        ).sum()

        add_result(
            "support_tickets",
            "customer_id",
            "Customer ID must exist in customers",
            invalid_customer,
            "Critical"
        )

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

    invalid_dates = (
        created.notna()
        & resolved.notna()
        & (resolved < created)
    ).sum()

    add_result(
        "support_tickets",
        "resolved_date",
        "Resolved date must not be before created date",
        invalid_dates,
        "High"
    )

    resolution_hours = pd.to_numeric(
        df["resolution_hours"],
        errors="coerce"
    )

    negative_hours = (
        resolution_hours.notna()
        & (resolution_hours < 0)
    ).sum()

    add_result(
        "support_tickets",
        "resolution_hours",
        "Resolution hours must not be negative",
        negative_hours,
        "High"
    )

    very_high_hours = (
        resolution_hours.notna()
        & (resolution_hours > 720)
    ).sum()

    add_result(
        "support_tickets",
        "resolution_hours",
        "Resolution time above 720 hours should be reviewed",
        very_high_hours,
        "Medium"
    )

    satisfaction = pd.to_numeric(
        df["satisfaction_score"],
        errors="coerce"
    )

    invalid_satisfaction = (
        satisfaction.notna()
        & (
            (satisfaction < 1)
            | (satisfaction > 5)
        )
    ).sum()

    add_result(
        "support_tickets",
        "satisfaction_score",
        "Satisfaction score must be between 1 and 5",
        invalid_satisfaction,
        "High"
    )


# ============================================================
# RUN QUALITY RULES
# ============================================================

if "customers" in dataframes:
    validate_customers(
        dataframes["customers"]
    )

if "products" in dataframes:
    validate_products(
        dataframes["products"]
    )

if "sales_representatives" in dataframes:
    validate_sales_representatives(
        dataframes["sales_representatives"]
    )

if "sales" in dataframes:
    validate_sales(
        dataframes["sales"],
        dataframes.get("customers"),
        dataframes.get("products"),
        dataframes.get("sales_representatives")
    )

if "leads" in dataframes:
    validate_leads(
        dataframes["leads"]
    )

if "opportunities" in dataframes:
    validate_opportunities(
        dataframes["opportunities"],
        dataframes.get("leads"),
        dataframes.get("customers"),
        dataframes.get("sales_representatives")
    )

if "campaigns" in dataframes:
    validate_campaigns(
        dataframes["campaigns"]
    )

if "campaign_interactions" in dataframes:
    validate_campaign_interactions(
        dataframes["campaign_interactions"],
        dataframes.get("campaigns"),
        dataframes.get("customers")
    )

if "support_tickets" in dataframes:
    validate_support_tickets(
        dataframes["support_tickets"],
        dataframes.get("customers")
    )


# ============================================================
# CREATE DATA QUALITY REPORT
# ============================================================

quality_df = pd.DataFrame(
    quality_results
)

quality_df["status"] = quality_df[
    "issue_count"
].apply(
    lambda x: "PASS"
    if x == 0
    else "FAIL"
)

quality_df = quality_df[
    [
        "table",
        "column",
        "rule",
        "issue_count",
        "severity",
        "status"
    ]
]

output_path = (
    QUALITY_DIR /
    "data_quality_rules_report.csv"
)

quality_df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 80)
print("DATA QUALITY RULES REPORT")
print("=" * 80)

print(quality_df.to_string(index=False))

print(
    f"\nReport saved to:\n{output_path}"
)

print(
    "\nDATA QUALITY RULES VALIDATION COMPLETED."
)