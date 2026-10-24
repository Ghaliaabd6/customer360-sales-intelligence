"""Generate reproducible, intentionally imperfect Customer 360 RAW data.

This module only creates synthetic CSV files and a data-quality injection report.
It does not clean, transform, load, or publish the data.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from faker import Faker
except ImportError:  # Faker is listed in requirements.txt but is optional at runtime.
    Faker = None


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
START_DATE = date(2021, 1, 1)
END_DATE = date(2025, 12, 31)

COUNTRIES = ["France", "United Kingdom", "Germany", "Spain", "Italy", "Belgium", "Netherlands"]
REGIONS = {
    "France": ["Ile-de-France", "Auvergne-Rhone-Alpes", "Provence-Alpes-Cote d'Azur"],
    "United Kingdom": ["England", "Scotland", "Wales"],
    "Germany": ["Bavaria", "Berlin", "North Rhine-Westphalia"],
    "Spain": ["Madrid", "Catalonia", "Andalusia"],
    "Italy": ["Lazio", "Lombardy", "Tuscany"],
    "Belgium": ["Brussels", "Flanders", "Wallonia"],
    "Netherlands": ["North Holland", "South Holland", "Utrecht"],
}
CITIES = {
    "France": ["Paris", "Lyon", "Marseille", "Toulouse"],
    "United Kingdom": ["London", "Manchester", "Edinburgh", "Cardiff"],
    "Germany": ["Berlin", "Munich", "Hamburg", "Cologne"],
    "Spain": ["Madrid", "Barcelona", "Seville", "Valencia"],
    "Italy": ["Rome", "Milan", "Florence", "Turin"],
    "Belgium": ["Brussels", "Antwerp", "Liege", "Ghent"],
    "Netherlands": ["Amsterdam", "Rotterdam", "Utrecht", "The Hague"],
}
FIRST_NAMES = ["Alex", "Camille", "Jordan", "Morgan", "Sofia", "Lucas", "Emma", "Noah", "Mia", "Hugo"]
LAST_NAMES = ["Martin", "Bernard", "Dubois", "Moreau", "Garcia", "Smith", "Muller", "Rossi", "Jansen", "Taylor"]


def random_date(rng: np.random.Generator, start: date = START_DATE, end: date = END_DATE) -> date:
    span = (end - start).days
    return start + timedelta(days=int(rng.integers(0, span + 1)))


def money(value: float) -> float:
    return round(float(value), 2)


def make_identity(fake: Any, index: int) -> tuple[str, str]:
    if fake is not None:
        return fake.first_name(), fake.last_name()
    return FIRST_NAMES[index % len(FIRST_NAMES)], LAST_NAMES[(index * 3) % len(LAST_NAMES)]


def build_customers(rng: np.random.Generator, fake: Any, n: int = 10_000) -> pd.DataFrame:
    rows = []
    for i in range(n):
        first, last = make_identity(fake, i)
        country = str(rng.choice(COUNTRIES))
        segment = str(rng.choice(["Consumer", "SMB", "Enterprise"], p=[0.68, 0.22, 0.10]))
        loyalty = str(rng.choice(["Bronze", "Silver", "Gold", "Platinum"], p=[0.42, 0.34, 0.18, 0.06]))
        signup = random_date(rng, date(2019, 1, 1), END_DATE)
        email = f"{first}.{last}.{i + 1}@example.test".lower()
        rows.append({
            "customer_id": f"CUST{i + 1:06d}", "first_name": first, "last_name": last,
            "email": email, "phone": f"+33 6 {int(rng.integers(10_000_000, 99_999_999))}",
            "birth_date": random_date(rng, date(1940, 1, 1), date(2004, 12, 31)),
            "gender": str(rng.choice(["Female", "Male", "Non-binary"])),
            "signup_date": signup, "country": country, "region": str(rng.choice(REGIONS[country])),
            "city": str(rng.choice(CITIES[country])), "customer_segment": segment,
            "acquisition_channel": str(rng.choice(
                ["Organic", "Paid Search", "Social Media", "Referral", "Partner", "Email"],
                p=[0.28, 0.18, 0.16, 0.16, 0.12, 0.10],
            )),
            "loyalty_level": loyalty, "is_active": bool(rng.random() < 0.84), "churn_flag": False,
        })
    return pd.DataFrame(rows)


def build_products(rng: np.random.Generator, n: int = 300) -> pd.DataFrame:
    categories = {
        "Technology": ["Laptops", "Monitors", "Accessories"],
        "Office": ["Furniture", "Supplies", "Storage"],
        "Services": ["Consulting", "Support", "Training"],
        "Security": ["Hardware", "Software", "Monitoring"],
    }
    rows = []
    for i in range(n):
        category = str(rng.choice(list(categories)))
        subcategory = str(rng.choice(categories[category]))
        cost = money(rng.uniform(8, 1_200))
        rows.append({
            "product_id": f"PROD{i + 1:04d}", "product_name": f"{subcategory} {i + 1:03d}",
            "category": category, "subcategory": subcategory, "unit_cost": cost,
            "unit_price": money(cost * rng.uniform(1.25, 3.4)),
            "launch_date": random_date(rng, date(2018, 1, 1), END_DATE),
            "status": str(rng.choice(["Active", "Discontinued", "Pre-Launch"], p=[0.86, 0.10, 0.04])),
        })
    return pd.DataFrame(rows)


def build_reps(rng: np.random.Generator, fake: Any, n: int = 80) -> pd.DataFrame:
    rows = []
    for i in range(n):
        first, last = make_identity(fake, i + 100)
        region = str(rng.choice(["North", "South", "East", "West", "Central"]))
        rows.append({
            "sales_rep_id": f"REP{i + 1:03d}", "first_name": first, "last_name": last,
            "email": f"{first}.{last}.rep{i + 1}@example.test".lower(), "region": region,
            "team": f"{region} Sales", "hire_date": random_date(rng, date(2015, 1, 1), date(2024, 12, 31)),
            "manager_name": f"Manager {int(rng.integers(1, 13)):02d}",
            "status": str(rng.choice(["Active", "On Leave", "Inactive"], p=[0.88, 0.04, 0.08])),
        })
    return pd.DataFrame(rows)


def build_sales(rng: np.random.Generator, customers: pd.DataFrame, products: pd.DataFrame,
                reps: pd.DataFrame, n: int = 60_000) -> pd.DataFrame:
    rows = []
    loyalty_factor = {"Bronze": 1.0, "Silver": 1.08, "Gold": 1.18, "Platinum": 1.32}
    segment_factor = {"Consumer": 1.0, "SMB": 1.7, "Enterprise": 3.6}
    for i in range(n):
        customer = customers.iloc[int(rng.integers(len(customers)))]
        product = products.iloc[int(rng.integers(len(products)))]
        quantity = int(rng.choice([1, 1, 1, 2, 2, 3, 4]))
        discount = float(rng.choice([0, 0.05, 0.10, 0.15, 0.20], p=[.25, .27, .28, .15, .05]))
        price = float(product.unit_price) * segment_factor[customer.customer_segment] ** 0.08
        gross = quantity * price * loyalty_factor[customer.loyalty_level]
        rows.append({
            "sale_id": f"SALE{i + 1:07d}", "order_id": f"ORD{(i // 1) + 1:07d}",
            "order_date": random_date(rng, date(2022, 1, 1), END_DATE),
            "customer_id": customer.customer_id, "product_id": product.product_id,
            "sales_rep_id": str(reps.iloc[int(rng.integers(len(reps)))].sales_rep_id),
            "quantity": quantity, "unit_price": money(price), "discount_pct": discount,
            "gross_amount": money(gross), "discount_amount": money(gross * discount),
            "net_amount": money(gross * (1 - discount)),
            "sales_channel": str(rng.choice(["Online", "Store", "Partner", "Phone"], p=[.48, .25, .17, .10])),
            "payment_method": str(rng.choice(["Credit Card", "Bank Transfer", "PayPal", "Cash", "Direct Debit"])),
            "region": str(customer.region),
        })
    return pd.DataFrame(rows)


def build_leads(rng: np.random.Generator, fake: Any, n: int = 15_000) -> pd.DataFrame:
    rows = []
    for i in range(n):
        first, last = make_identity(fake, i + 300)
        country = str(rng.choice(COUNTRIES))
        score = int(np.clip(rng.normal(58, 22), 1, 100))
        status = str(rng.choice(["New", "Contacted", "Qualified", "Disqualified", "Converted"],
                                p=[.22, .24, .20, .18, .16]))
        if score >= 80 and rng.random() < .55:
            status = "Converted"
        rows.append({
            "lead_id": f"LEAD{i + 1:06d}", "created_date": random_date(rng, date(2021, 1, 1), END_DATE),
            "first_name": first, "last_name": last, "email": f"{first}.{last}.lead{i + 1}@example.test".lower(),
            "company": f"Company {i + 1:05d}", "country": country, "region": str(rng.choice(REGIONS[country])),
            "source": str(rng.choice(["Website", "LinkedIn", "Campaign", "Referral", "Event", "Partner"])),
            "industry": str(rng.choice(["Technology", "Finance", "Retail", "Healthcare", "Manufacturing"])),
            "company_size": str(rng.choice(["1-10", "11-50", "51-250", "251-1000", "1000+"])),
            "lead_score": score, "lead_status": status,
        })
    return pd.DataFrame(rows)


def build_opportunities(rng: np.random.Generator, leads: pd.DataFrame, customers: pd.DataFrame,
                        reps: pd.DataFrame) -> pd.DataFrame:
    chosen = leads.sample(n=6_000, random_state=SEED).reset_index(drop=True)
    rows = []
    stages = ["Discovery", "Qualification", "Proposal", "Negotiation", "Closed Won", "Closed Lost"]
    probabilities = {"Discovery": .15, "Qualification": .30, "Proposal": .55, "Negotiation": .75,
                     "Closed Won": 1.0, "Closed Lost": 0.0}
    for i, lead in chosen.iterrows():
        created = lead.created_date
        stage = str(rng.choice(stages, p=[.20, .19, .18, .14, .20, .09]))
        won = stage == "Closed Won"
        customer_id = str(customers.iloc[int(rng.integers(len(customers)))].customer_id) if won else None
        expected = money(rng.uniform(1_000, 150_000) * (1 + lead.lead_score / 250))
        actual = money(expected * rng.uniform(.80, 1.15)) if won else None
        rows.append({
            "opportunity_id": f"OPP{i + 1:06d}", "lead_id": lead.lead_id, "customer_id": customer_id,
            "sales_rep_id": str(reps.iloc[int(rng.integers(len(reps)))].sales_rep_id),
            "created_date": created, "expected_close_date": created + timedelta(days=int(rng.integers(20, 240))),
            "actual_close_date": created + timedelta(days=int(rng.integers(20, 240))) if stage in ["Closed Won", "Closed Lost"] else None,
            "stage": stage, "probability": probabilities[stage], "expected_revenue": expected,
            "actual_revenue": actual, "status": "Open" if stage not in ["Closed Won", "Closed Lost"] else stage,
        })
    return pd.DataFrame(rows)


def build_campaigns(rng: np.random.Generator, n: int = 120) -> pd.DataFrame:
    rows = []
    for i in range(n):
        start = random_date(rng, date(2022, 1, 1), date(2025, 9, 1))
        rows.append({
            "campaign_id": f"CAMP{i + 1:04d}", "campaign_name": f"Campaign {i + 1:03d}",
            "campaign_type": str(rng.choice(["Acquisition", "Retention", "Upsell", "Cross-sell", "Reactivation"])),
            "channel": str(rng.choice(["Email", "Social Media", "Paid Search", "Display", "Partner"])),
            "start_date": start, "end_date": start + timedelta(days=int(rng.integers(7, 60))),
            "budget": money(rng.uniform(5_000, 250_000)), "target_segment": str(rng.choice(["Consumer", "SMB", "Enterprise"])),
            "region": str(rng.choice(["North", "South", "East", "West", "Global"])),
            "status": str(rng.choice(["Planned", "Active", "Completed"], p=[.12, .23, .65])),
        })
    return pd.DataFrame(rows)


def build_interactions(rng: np.random.Generator, campaigns: pd.DataFrame,
                       customers: pd.DataFrame, n: int = 40_000) -> pd.DataFrame:
    rows = []
    for i in range(n):
        opened = bool(rng.random() < .58)
        clicked = bool(opened and rng.random() < .36)
        converted = bool(clicked and rng.random() < .14)
        rows.append({
            "interaction_id": f"INT{i + 1:07d}", "campaign_id": str(campaigns.iloc[int(rng.integers(len(campaigns)))].campaign_id),
            "customer_id": str(customers.iloc[int(rng.integers(len(customers)))].customer_id),
            "interaction_date": random_date(rng, date(2022, 1, 1), END_DATE),
            "channel": str(rng.choice(["Email", "Social Media", "Paid Search", "Display", "Partner"])),
            "opened": opened, "clicked": clicked, "converted": converted,
            "conversion_value": money(rng.uniform(100, 20_000)) if converted else 0.0,
        })
    return pd.DataFrame(rows)


def build_tickets(rng: np.random.Generator, customers: pd.DataFrame, n: int = 18_000) -> pd.DataFrame:
    rows = []
    for i in range(n):
        created = random_date(rng, date(2022, 1, 1), END_DATE)
        priority = str(rng.choice(["Low", "Medium", "High", "Critical"], p=[.35, .40, .20, .05]))
        status = str(rng.choice(["Open", "Pending", "Resolved", "Closed"], p=[.10, .15, .35, .40]))
        hours = float(rng.gamma(2.2, {"Low": 8, "Medium": 18, "High": 32, "Critical": 65}[priority]))
        resolved = created + timedelta(hours=int(hours)) if status in ["Resolved", "Closed"] else None
        rows.append({
            "ticket_id": f"TICK{i + 1:06d}", "customer_id": str(customers.iloc[int(rng.integers(len(customers)))].customer_id),
            "created_date": created, "resolved_date": resolved,
            "category": str(rng.choice(["Billing", "Technical", "Account", "Delivery", "Product", "Complaint"])),
            "priority": priority, "channel": str(rng.choice(["Email", "Phone", "Chat", "Web"])),
            "status": status, "resolution_hours": round(hours, 2) if resolved else None,
            "satisfaction_score": int(rng.integers(1, 6)) if status in ["Resolved", "Closed"] else None,
            "agent_team": str(rng.choice(["Tier 1", "Tier 2", "Billing Desk", "Escalations"])),
        })
    return pd.DataFrame(rows)


def assign_churn_flags(customers: pd.DataFrame, tickets: pd.DataFrame, rng: np.random.Generator) -> None:
    ticket_counts = tickets.groupby("customer_id").size()
    low_satisfaction = (
        tickets.assign(low_score=tickets["satisfaction_score"].fillna(3) <= 2)
        .groupby("customer_id")["low_score"]
        .sum()
    )
    loyalty_risk = customers["loyalty_level"].map(
        {"Bronze": 0.07, "Silver": 0.045, "Gold": 0.025, "Platinum": 0.015}
    ).fillna(0.05)
    ticket_risk = customers["customer_id"].map(ticket_counts).fillna(0).clip(upper=8) * 0.012
    satisfaction_risk = customers["customer_id"].map(low_satisfaction).fillna(0).clip(upper=4) * 0.025
    inactivity_risk = (~customers["is_active"]).astype(float) * 0.25
    probability = np.clip(loyalty_risk + ticket_risk + satisfaction_risk + inactivity_risk, 0, 0.95)
    customers["churn_flag"] = rng.random(len(customers)) < probability.to_numpy()


def inject_anomalies(frames: dict[str, pd.DataFrame], rng: np.random.Generator) -> dict[str, Any]:
    report: dict[str, Any] = {name: {"rows": len(frame), "missing_values_injected": 0,
                                     "duplicates_injected": 0, "outliers_injected": 0,
                                     "categorical_inconsistencies": 0, "invalid_references": 0}
                               for name, frame in frames.items()}

    def set_value(dataset: str, row: int, column: str, value: Any) -> None:
        frames[dataset].at[row, column] = value

    # Missing values: a small, visible sample across business-critical columns.
    missing = {
        "customers": ["email", "phone", "region"], "sales": ["unit_price", "net_amount"],
        "leads": ["email"], "campaign_interactions": ["conversion_value"],
        "support_tickets": ["satisfaction_score", "resolved_date"],
    }
    for dataset, columns in missing.items():
        for column in columns:
            count = max(1, int(len(frames[dataset]) * 0.006))
            for row in rng.choice(len(frames[dataset]), size=count, replace=False):
                set_value(dataset, int(row), column, np.nan)
            report[dataset]["missing_values_injected"] += count

    # Exact row duplicates.
    for dataset, count in [("customers", 25), ("sales", 75), ("leads", 20)]:
        duplicate_rows = frames[dataset].sample(n=count, random_state=SEED + count)
        frames[dataset] = pd.concat([frames[dataset], duplicate_rows], ignore_index=True)
        report[dataset]["duplicates_injected"] = count
        report[dataset]["rows"] = len(frames[dataset])

    # Text casing/whitespace variants.
    for dataset, column, variants, count in [
        ("customers", "country", ["FRANCE", "france", " France"], 120),
        ("sales", "payment_method", ["credit card", "CREDIT CARD"], 100),
        ("campaign_interactions", "channel", ["email", "EMAIL"], 80),
        ("support_tickets", "category", ["technical", "TECH", "Tech Support"], 100),
    ]:
        for row in rng.choice(len(frames[dataset]), size=count, replace=False):
            set_value(dataset, int(row), column, str(rng.choice(variants)))
        report[dataset]["categorical_inconsistencies"] += count

    # Invalid email shapes and mixed date representations.
    for dataset, count in [("customers", 45), ("leads", 40)]:
        for row in rng.choice(len(frames[dataset]), size=count, replace=False):
            set_value(dataset, int(row), "email", str(rng.choice(["john.example.com", "john@", "@example.com"])))
        report[dataset]["categorical_inconsistencies"] += count
    for row in rng.choice(len(frames["customers"]), size=35, replace=False):
        value = frames["customers"].at[int(row), "signup_date"]
        set_value("customers", int(row), "signup_date", value.strftime("%d/%m/%Y"))
    report["customers"]["categorical_inconsistencies"] += 35

    # Numeric/date outliers and returns.
    for row in rng.choice(len(frames["sales"]), size=35, replace=False):
        set_value("sales", int(row), "quantity", int(rng.choice([99, 250, -3, -12])))
        report["sales"]["outliers_injected"] += 1
    for row in rng.choice(len(frames["sales"]), size=20, replace=False):
        set_value("sales", int(row), "unit_price", 999_999.99)
        report["sales"]["outliers_injected"] += 1
    for row in rng.choice(len(frames["sales"]), size=20, replace=False):
        set_value("sales", int(row), "discount_pct", float(rng.choice([0.75, 0.90, 1.25])))
        report["sales"]["outliers_injected"] += 1
    for row in rng.choice(len(frames["support_tickets"]), size=25, replace=False):
        set_value("support_tickets", int(row), "resolution_hours", 9_999.0)
        report["support_tickets"]["outliers_injected"] += 1
    for row in rng.choice(len(frames["support_tickets"]), size=15, replace=False):
        created = frames["support_tickets"].at[int(row), "created_date"]
        set_value("support_tickets", int(row), "resolved_date", created - timedelta(days=3))
        report["support_tickets"]["outliers_injected"] += 1

    # A very small referential-integrity challenge.
    for dataset, column, bad_values, count in [
        ("sales", "customer_id", ["CUST999999"], 50),
        ("sales", "product_id", ["PROD9999"], 35),
        ("campaign_interactions", "campaign_id", ["CAMP9999"], 20),
        ("campaign_interactions", "customer_id", ["CUST999999"], 20),
    ]:
        for row in rng.choice(len(frames[dataset]), size=count, replace=False):
            set_value(dataset, int(row), column, str(rng.choice(bad_values)))
        report[dataset]["invalid_references"] += count

    report["totals"] = {
        key: sum(item[key] for item in report.values() if isinstance(item, dict))
        for key in ["missing_values_injected", "duplicates_injected", "outliers_injected",
                    "categorical_inconsistencies", "invalid_references"]
    }
    return report


def write_and_validate(frames: dict[str, pd.DataFrame], report: dict[str, Any]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        path = RAW_DIR / f"{name}.csv"
        frame.to_csv(path, index=False, date_format="%Y-%m-%d")
        print(f"{name:<24} {len(frame):>7,} rows | {path.stat().st_size:>10,} bytes")
        print(f"  columns: {', '.join(frame.columns)}")

    report_path = RAW_DIR / "data_quality_injection_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    checks = {
        "sales.customer_id": set(frames["sales"]["customer_id"].dropna()) & set(frames["customers"]["customer_id"]),
        "sales.product_id": set(frames["sales"]["product_id"].dropna()) & set(frames["products"]["product_id"]),
        "sales.sales_rep_id": set(frames["sales"]["sales_rep_id"].dropna()) & set(frames["sales_representatives"]["sales_rep_id"]),
        "interactions.campaign_id": set(frames["campaign_interactions"]["campaign_id"].dropna()) & set(frames["campaigns"]["campaign_id"]),
    }
    print("\nRelation checks (matching referenced values):")
    for name, values in checks.items():
        print(f"  {name:<28} {len(values):>6,}")
    print(f"\nQuality anomalies injected: {json.dumps(report['totals'])}")
    print(f"Quality report: {report_path}")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    fake = Faker() if Faker is not None else None
    if fake is not None:
        fake.seed_instance(SEED)
    rng = np.random.default_rng(SEED)

    customers = build_customers(rng, fake)
    products = build_products(rng)
    reps = build_reps(rng, fake)
    frames = {
        "customers": customers,
        "products": products,
        "sales_representatives": reps,
        "sales": build_sales(rng, customers, products, reps),
        "leads": build_leads(rng, fake),
        "opportunities": None,
        "campaigns": build_campaigns(rng),
        "campaign_interactions": None,
        "support_tickets": build_tickets(rng, customers),
    }
    assign_churn_flags(customers, frames["support_tickets"], rng)
    frames["opportunities"] = build_opportunities(rng, frames["leads"], customers, reps)
    frames["campaign_interactions"] = build_interactions(rng, frames["campaigns"], customers)
    report = inject_anomalies(frames, rng)
    write_and_validate(frames, report)


if __name__ == "__main__":
    main()
