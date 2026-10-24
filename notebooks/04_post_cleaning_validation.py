from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


TABLES = [
    "customers",
    "products",
    "sales_representatives",
    "sales",
    "leads",
    "opportunities",
    "campaigns",
    "campaign_interactions",
    "support_tickets",
]


PRIMARY_KEYS = {
    "customers": "customer_id",
    "products": "product_id",
    "sales_representatives": "sales_rep_id",
    "sales": "sale_id",
    "leads": "lead_id",
    "opportunities": "opportunity_id",
    "campaigns": "campaign_id",
    "campaign_interactions": "interaction_id",
    "support_tickets": "ticket_id",
}


FOREIGN_KEYS = {
    "sales": [
        ("customer_id", "customers", "customer_id"),
        ("product_id", "products", "product_id"),
        ("sales_rep_id", "sales_representatives", "sales_rep_id"),
    ],
    "opportunities": [
        ("customer_id", "customers", "customer_id"),
        ("sales_rep_id", "sales_representatives", "sales_rep_id"),
    ],
    "campaign_interactions": [
        ("campaign_id", "campaigns", "campaign_id"),
        ("customer_id", "customers", "customer_id"),
    ],
    "support_tickets": [
        ("customer_id", "customers", "customer_id"),
    ],
}


def find_table_file(table_name):
    candidates = [
        PROCESSED_DIR / f"{table_name}.csv",
        PROCESSED_DIR / f"{table_name}_cleaned.csv",
        PROCESSED_DIR / f"cleaned_{table_name}.csv",
        PROCESSED_DIR / f"{table_name}_processed.csv",
    ]

    for file_path in candidates:
        if file_path.exists():
            return file_path

    matches = list(PROCESSED_DIR.glob(f"*{table_name}*.csv"))
    return matches[0] if matches else None


def load_tables():
    dataframes = {}

    for table in TABLES:
        file_path = find_table_file(table)

        if file_path is None:
            print(f"[ERROR] Missing processed file for: {table}")
            continue

        try:
            df = pd.read_csv(file_path, low_memory=False)
            dataframes[table] = df

            print(
                f"[OK] {table:<25} "
                f"{len(df):>7} rows | "
                f"{len(df.columns):>3} columns"
            )

        except Exception as exc:
            print(f"[ERROR] Failed to load {table}: {exc}")

    return dataframes


def add_result(results, table, check, status, issue_count=0, details=""):
    results.append(
        {
            "table": table,
            "check": check,
            "status": status,
            "issue_count": int(issue_count),
            "details": details,
        }
    )


def validate_tables_not_empty(dataframes, results):
    for table, df in dataframes.items():
        if df.empty:
            add_result(
                results,
                table,
                "table_not_empty",
                "FAIL",
                1,
                "Table contains no rows",
            )
        else:
            add_result(
                results,
                table,
                "table_not_empty",
                "PASS",
                0,
                f"{len(df)} rows",
            )


def validate_duplicates(dataframes, results):
    for table, df in dataframes.items():
        duplicate_count = int(df.duplicated().sum())

        add_result(
            results,
            table,
            "duplicate_rows",
            "PASS" if duplicate_count == 0 else "FAIL",
            duplicate_count,
            "Complete duplicate rows",
        )


def validate_primary_keys(dataframes, results):
    for table, primary_key in PRIMARY_KEYS.items():
        if table not in dataframes:
            continue

        df = dataframes[table]

        if primary_key not in df.columns:
            add_result(
                results,
                table,
                f"primary_key_{primary_key}",
                "FAIL",
                1,
                "Primary key column missing",
            )
            continue

        null_count = int(df[primary_key].isna().sum())
        duplicate_count = int(df[primary_key].duplicated().sum())
        issue_count = null_count + duplicate_count

        add_result(
            results,
            table,
            f"primary_key_{primary_key}",
            "PASS" if issue_count == 0 else "FAIL",
            issue_count,
            f"null={null_count}; duplicates={duplicate_count}",
        )


def validate_missing_values(dataframes, results):
    for table, df in dataframes.items():
        missing = df.isna().sum()
        missing = missing[missing > 0]

        total_missing = int(missing.sum())

        details = "; ".join(
            f"{column}={count}"
            for column, count in missing.items()
        )

        add_result(
            results,
            table,
            "missing_values",
            "PASS" if total_missing == 0 else "WARNING",
            total_missing,
            details or "No missing values",
        )


def validate_foreign_keys(dataframes, results):
    for child_table, relationships in FOREIGN_KEYS.items():
        if child_table not in dataframes:
            continue

        child_df = dataframes[child_table]

        for foreign_key, parent_table, parent_key in relationships:
            if parent_table not in dataframes:
                add_result(
                    results,
                    child_table,
                    f"foreign_key_{foreign_key}",
                    "WARNING",
                    1,
                    f"Parent table {parent_table} unavailable",
                )
                continue

            parent_df = dataframes[parent_table]

            if foreign_key not in child_df.columns:
                add_result(
                    results,
                    child_table,
                    f"foreign_key_{foreign_key}",
                    "FAIL",
                    1,
                    f"Column {foreign_key} missing",
                )
                continue

            if parent_key not in parent_df.columns:
                add_result(
                    results,
                    child_table,
                    f"foreign_key_{foreign_key}",
                    "FAIL",
                    1,
                    f"Parent key {parent_key} missing",
                )
                continue

            child_values = child_df[foreign_key].dropna()
            parent_values = set(parent_df[parent_key].dropna())

            invalid_count = int(
                (~child_values.isin(parent_values)).sum()
            )

            add_result(
                results,
                child_table,
                f"foreign_key_{foreign_key}",
                "PASS" if invalid_count == 0 else "FAIL",
                invalid_count,
                f"Reference: {parent_table}.{parent_key}",
            )


def validate_dates(dataframes, results):
    for table, df in dataframes.items():
        date_columns = [
            column
            for column in df.columns
            if "date" in column.lower()
            or column.lower().endswith("_at")
        ]

        for column in date_columns:
            values = df[column].dropna()

            if values.empty:
                continue

            parsed = pd.to_datetime(values, errors="coerce")
            invalid_count = int(parsed.isna().sum())

            add_result(
                results,
                table,
                f"date_format_{column}",
                "PASS" if invalid_count == 0 else "FAIL",
                invalid_count,
                "Date parsing validation",
            )


def validate_numeric_values(dataframes, results):
    keywords = [
        "amount",
        "price",
        "quantity",
        "revenue",
        "cost",
        "budget",
        "duration",
    ]

    for table, df in dataframes.items():
        numeric_columns = df.select_dtypes(include=np.number).columns

        for column in numeric_columns:
            if not any(
                keyword in column.lower()
                for keyword in keywords
            ):
                continue

            negative_count = int(
                (df[column].dropna() < 0).sum()
            )

            add_result(
                results,
                table,
                f"negative_values_{column}",
                "PASS" if negative_count == 0 else "WARNING",
                negative_count,
                "Values expected to be non-negative",
            )


def validate_business_rules(dataframes, results):
    if "sales" in dataframes:
        sales = dataframes["sales"]

        if "quantity" in sales.columns:
            quantity = pd.to_numeric(
                sales["quantity"],
                errors="coerce",
            )

            zero_count = int((quantity == 0).sum())
            negative_count = int((quantity < 0).sum())

            add_result(
                results,
                "sales",
                "quantity_non_zero",
                "PASS" if zero_count == 0 else "FAIL",
                zero_count,
                "Quantity must not be zero",
            )

            add_result(
                results,
                "sales",
                "negative_quantity_returns",
                "WARNING" if negative_count > 0 else "PASS",
                negative_count,
                "Negative quantities may represent returns or cancellations",
            )

        for column in [
            "sales_amount",
            "total_amount",
            "revenue",
            "amount",
        ]:
            if column not in sales.columns:
                continue

            values = pd.to_numeric(
                sales[column],
                errors="coerce",
            )

            invalid_count = int((values < 0).sum())

            add_result(
                results,
                "sales",
                f"{column}_non_negative",
                "PASS" if invalid_count == 0 else "FAIL",
                invalid_count,
                "Sales amount must be non-negative",
            )

    if "products" in dataframes:
        products = dataframes["products"]

        for column in [
            "price",
            "unit_price",
            "selling_price",
        ]:
            if column not in products.columns:
                continue

            values = pd.to_numeric(
                products[column],
                errors="coerce",
            )

            invalid_count = int((values < 0).sum())

            add_result(
                results,
                "products",
                f"{column}_non_negative",
                "PASS" if invalid_count == 0 else "FAIL",
                invalid_count,
                "Product price must be non-negative",
            )

    if "opportunities" in dataframes:
        opportunities = dataframes["opportunities"]

        for column in [
            "probability",
            "win_probability",
        ]:
            if column not in opportunities.columns:
                continue

            values = pd.to_numeric(
                opportunities[column],
                errors="coerce",
            )

            maximum = values.max()

            if pd.notna(maximum) and maximum <= 1:
                invalid_count = int(
                    ((values < 0) | (values > 1)).sum()
                )
                expected_range = "0 to 1"
            else:
                invalid_count = int(
                    ((values < 0) | (values > 100)).sum()
                )
                expected_range = "0 to 100"

            add_result(
                results,
                "opportunities",
                f"{column}_range",
                "PASS" if invalid_count == 0 else "FAIL",
                invalid_count,
                f"Expected range: {expected_range}",
            )

    if "campaigns" in dataframes:
        campaigns = dataframes["campaigns"]

        if {
            "start_date",
            "end_date",
        }.issubset(campaigns.columns):

            start_date = pd.to_datetime(
                campaigns["start_date"],
                errors="coerce",
            )

            end_date = pd.to_datetime(
                campaigns["end_date"],
                errors="coerce",
            )

            invalid_count = int(
                (end_date < start_date).sum()
            )

            add_result(
                results,
                "campaigns",
                "campaign_date_consistency",
                "PASS" if invalid_count == 0 else "FAIL",
                invalid_count,
                "End date must not precede start date",
            )


def generate_dataset_summary(dataframes):
    rows = []

    for table, df in dataframes.items():
        total_cells = df.shape[0] * df.shape[1]
        missing_values = int(df.isna().sum().sum())

        missing_percentage = (
            missing_values / total_cells * 100
            if total_cells
            else 0
        )

        rows.append(
            {
                "table": table,
                "rows": len(df),
                "columns": len(df.columns),
                "missing_values": missing_values,
                "missing_percentage": round(
                    missing_percentage,
                    2,
                ),
                "duplicate_rows": int(
                    df.duplicated().sum()
                ),
            }
        )

    summary_df = pd.DataFrame(rows)

    output_path = (
        REPORTS_DIR
        / "post_cleaning_dataset_summary.csv"
    )

    summary_df.to_csv(
        output_path,
        index=False,
    )

    return summary_df


def save_validation_report(results):
    report_df = pd.DataFrame(results)

    if report_df.empty:
        return report_df

    status_order = {
        "FAIL": 0,
        "WARNING": 1,
        "PASS": 2,
    }

    report_df["status_order"] = (
        report_df["status"]
        .map(status_order)
        .fillna(99)
    )

    report_df = report_df.sort_values(
        ["status_order", "table", "check"]
    ).drop(columns=["status_order"])

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    timestamped_path = (
        REPORTS_DIR
        / f"post_cleaning_validation_{timestamp}.csv"
    )

    latest_path = (
        REPORTS_DIR
        / "post_cleaning_validation_latest.csv"
    )

    report_df.to_csv(
        timestamped_path,
        index=False,
    )

    report_df.to_csv(
        latest_path,
        index=False,
    )

    return report_df


def print_summary(report_df):
    if report_df.empty:
        print("No validation results generated.")
        return

    counts = (
        report_df["status"]
        .value_counts()
        .to_dict()
    )

    pass_count = counts.get("PASS", 0)
    warning_count = counts.get("WARNING", 0)
    fail_count = counts.get("FAIL", 0)

    print("\nPost-cleaning validation")
    print("-" * 40)
    print(f"PASS    : {pass_count}")
    print(f"WARNING : {warning_count}")
    print(f"FAIL    : {fail_count}")

    if fail_count == 0:
        print("\nProcessed data passed critical validation.")
    else:
        print(
            "\nCritical validation issues detected. "
            "Review the validation report before loading data."
        )


def main():
    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(
            f"Processed directory not found: {PROCESSED_DIR}"
        )

    print("Loading processed datasets...\n")

    dataframes = load_tables()

    if not dataframes:
        raise RuntimeError(
            "No processed datasets were loaded."
        )

    results = []

    validate_tables_not_empty(dataframes, results)
    validate_duplicates(dataframes, results)
    validate_primary_keys(dataframes, results)
    validate_missing_values(dataframes, results)
    validate_foreign_keys(dataframes, results)
    validate_dates(dataframes, results)
    validate_numeric_values(dataframes, results)
    validate_business_rules(dataframes, results)

    generate_dataset_summary(dataframes)

    report_df = save_validation_report(results)

    print_summary(report_df)


if __name__ == "__main__":
    main()