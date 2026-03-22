"""Generate synthetic golden dataset for multi-hop constraint satisfaction evaluation.

Dataset design:
  - 5 planted TRUE violations (gold = these row IDs)
  - 4 near-miss rows: satisfy N-1 of N conditions in compound rules → NOT violations
    Baseline tends to flag near-misses (false positives); ReAct checks all conditions.
  - 6 clearly compliant rows

This structure forces the model to check ALL conditions of compound rules.
Baseline failure mode: flags near-misses → low precision.
ReAct advantage: EVALUATE step checks each condition explicitly → fewer false positives.

Usage:
    python scripts/generate_data.py
"""

import json
import random
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "constraints_golden.json"
RNG = random.Random(42)


def fmt_csv(headers: list[str], rows: list[dict]) -> str:
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(row[h]) for h in headers))
    return "\n".join(lines)


def bv(b: bool) -> str:
    return "true" if b else "false"


def difficulty_for(violated: list[str], hard: set[str], medium: set[str]) -> str:
    if any(r in hard for r in violated): return "hard"
    if any(r in medium for r in violated): return "medium"
    return "easy"


# ---------------------------------------------------------------------------
# Category 1: order_compliance
# ---------------------------------------------------------------------------

def make_order_compliance(seed: int, variant: int) -> dict:
    RNG.seed(seed)
    vt = [300, 400, 500, 600, 250, 450, 350, 500, 550, 400][variant % 10]

    rules = [
        f"Rule-A: Orders from unverified customers (customer_verified=false) with order_value > {vt} must have manual_review=true.",
        "Rule-B: Express shipping (shipping_type=express) is prohibited when the order has fragile items (has_fragile=true) AND the destination is international (is_international=true).",
        "Rule-C: A discount code (discount_applied=true) cannot be used on orders that already have a promotional price (has_promo_price=true).",
        "Rule-D: Gift wrapping (gift_wrap=true) is not available for orders with order_value < 50 OR for digital-only orders (is_digital=true).",
    ]

    def vA(r): return not r["customer_verified"] and r["order_value"] > vt and not r["manual_review"]
    def vB(r): return r["shipping_type"] == "express" and r["has_fragile"] and r["is_international"]
    def vC(r): return r["discount_applied"] and r["has_promo_price"]
    def vD(r): return r["gift_wrap"] and (r["order_value"] < 50 or r["is_digital"])

    def base(oid):
        return dict(order_id=oid, customer_verified=True, order_value=200.0,
                    shipping_type="standard", has_fragile=False, is_international=False,
                    has_promo_price=False, discount_applied=False,
                    gift_wrap=False, is_digital=False, manual_review=False)

    rows = [base(f"ORD-{i+1:03d}") for i in range(15)]

    # TRUE violations (gold)
    rows[0].update(dict(customer_verified=False, order_value=vt+100, manual_review=False))   # Rule-A easy
    rows[2].update(dict(shipping_type="express", has_fragile=True, is_international=True))    # Rule-B medium
    rows[4].update(dict(discount_applied=True, has_promo_price=True))                         # Rule-C easy
    rows[6].update(dict(gift_wrap=True, order_value=35.0, is_digital=False))                  # Rule-D hard (OR branch 1)
    rows[8].update(dict(gift_wrap=True, is_digital=True, order_value=150.0))                  # Rule-D hard (OR branch 2)

    # NEAR-MISSES (NOT violations — baseline tends to flag these)
    # Near-miss Rule-B: express + fragile but NOT international
    rows[10].update(dict(shipping_type="express", has_fragile=True, is_international=False))
    # Near-miss Rule-B: express + international but NOT fragile
    rows[11].update(dict(shipping_type="express", has_fragile=False, is_international=True))
    # Near-miss Rule-A: unverified but order_value exactly AT threshold (not above)
    rows[12].update(dict(customer_verified=False, order_value=float(vt), manual_review=False))
    # Near-miss Rule-D: gift_wrap=true, order_value>=50, is_digital=false → no violation
    rows[13].update(dict(gift_wrap=True, order_value=75.0, is_digital=False))

    gold = []
    for row in rows:
        v = [n for n, f in [("Rule-A",vA),("Rule-B",vB),("Rule-C",vC),("Rule-D",vD)] if f(row)]
        if v:
            gold.append({"row_id": row["order_id"], "rules_violated": v,
                         "difficulty": difficulty_for(v, {"Rule-D"}, {"Rule-B"})})

    headers = ["order_id","customer_verified","order_value","shipping_type",
               "has_fragile","is_international","has_promo_price",
               "discount_applied","gift_wrap","is_digital","manual_review"]
    csv_rows = [{h: bv(v) if isinstance(v, bool) else v for h,v in row.items()} for row in rows]
    return {"category":"order_compliance","rules":rules,
            "data_csv":fmt_csv(headers,csv_rows),"gold_violations":gold}


# ---------------------------------------------------------------------------
# Category 2: hr_policy
# ---------------------------------------------------------------------------

def make_hr_policy(seed: int, variant: int) -> dict:
    RNG.seed(seed)
    pip_score = [2.5,2.0,3.0,2.5,2.0,3.0,2.5,2.0,2.5,3.0][variant%10]
    pip_tenure = [1.0,1.5,1.0,2.0,1.0,1.5,1.0,2.0,1.5,1.0][variant%10]
    mgr_reports = [8,6,10,8,7,9,8,6,10,8][variant%10]
    hw_budget = [2000,1500,2500,2000,1800,2200,2000,1500,2500,2000][variant%10]

    rules = [
        f"Rule-A: Employees with performance_score < {pip_score} AND tenure_years < {pip_tenure} must have pip_required=true.",
        f"Rule-B: Employees with role=manager AND direct_reports > {mgr_reports} must have management_cert=true.",
        f"Rule-C: Remote employees (is_remote=true) outside Engineering cannot have hardware_budget > {hw_budget}.",
        "Rule-D: Employees with attrition_risk=high who have NOT received a retention action (retention_action=false) must have needs_escalation=true.",
    ]

    def vA(r): return r["performance_score"] < pip_score and r["tenure_years"] < pip_tenure and not r["pip_required"]
    def vB(r): return r["role"]=="manager" and r["direct_reports"] > mgr_reports and not r["management_cert"]
    def vC(r): return r["is_remote"] and r["department"]!="Engineering" and r["hardware_budget"] > hw_budget
    def vD(r): return r["attrition_risk"]=="high" and not r["retention_action"] and not r["needs_escalation"]

    def base(eid):
        return dict(emp_id=eid, department="Sales", role="individual",
                    tenure_years=3.0, performance_score=4.0, is_remote=False,
                    direct_reports=0, management_cert=True,
                    hardware_budget=hw_budget-500,
                    pip_required=False, attrition_risk="low",
                    retention_action=True, needs_escalation=False)

    rows = [base(f"EMP-{i+1:03d}") for i in range(15)]

    # TRUE violations
    rows[0].update(dict(performance_score=pip_score-0.4, tenure_years=pip_tenure-0.3, pip_required=False))  # Rule-A easy
    rows[2].update(dict(role="manager", direct_reports=mgr_reports+3, management_cert=False))                # Rule-B medium
    rows[4].update(dict(is_remote=True, department="Sales", hardware_budget=hw_budget+400))                  # Rule-C medium
    rows[6].update(dict(attrition_risk="high", retention_action=False, needs_escalation=False))              # Rule-D hard
    rows[8].update(dict(attrition_risk="high", retention_action=False, needs_escalation=False))              # Rule-D hard

    # NEAR-MISSES
    # Near-miss Rule-A: low performance but tenure >= threshold (NOT violation)
    rows[10].update(dict(performance_score=pip_score-0.3, tenure_years=pip_tenure+0.5, pip_required=False))
    # Near-miss Rule-B: manager but direct_reports exactly AT threshold (not above)
    rows[11].update(dict(role="manager", direct_reports=mgr_reports, management_cert=False))
    # Near-miss Rule-C: remote but Engineering department (exempt)
    rows[12].update(dict(is_remote=True, department="Engineering", hardware_budget=hw_budget+500))
    # Near-miss Rule-D: attrition_risk=high but retention_action=true (action taken)
    rows[13].update(dict(attrition_risk="high", retention_action=True, needs_escalation=False))

    gold = []
    for row in rows:
        v = [n for n,f in [("Rule-A",vA),("Rule-B",vB),("Rule-C",vC),("Rule-D",vD)] if f(row)]
        if v:
            gold.append({"row_id":row["emp_id"],"rules_violated":v,
                         "difficulty":difficulty_for(v,{"Rule-D"},{"Rule-B","Rule-C"})})

    headers = ["emp_id","department","role","tenure_years","performance_score",
               "is_remote","direct_reports","management_cert","hardware_budget",
               "pip_required","attrition_risk","retention_action","needs_escalation"]
    csv_rows = [{h:bv(v) if isinstance(v,bool) else v for h,v in row.items()} for row in rows]
    return {"category":"hr_policy","rules":rules,
            "data_csv":fmt_csv(headers,csv_rows),"gold_violations":gold}


# ---------------------------------------------------------------------------
# Category 3: inventory_rules
# ---------------------------------------------------------------------------

def make_inventory_rules(seed: int, variant: int) -> dict:
    RNG.seed(seed)
    expiry_days = [7,5,10,7,5,10,7,5,10,7][variant%10]
    overstock_qty = [200,150,250,200,150,250,200,150,250,200][variant%10]
    markup = [1.5,2.0,1.8,1.5,2.0,1.8,1.5,2.0,1.8,1.5][variant%10]

    rules = [
        "Rule-A: Items where stock_level <= reorder_point must have reorder_pending=true.",
        f"Rule-B: Perishable items (is_perishable=true) with days_until_expiry <= {expiry_days} AND stock_level > {overstock_qty} must have markdown_flagged=true.",
        f"Rule-C: Items where selling_price < cost_price * {markup} must have pricing_alert=true.",
        "Rule-D: Hazardous items (is_hazardous=true) in a non-compliant location (location_compliant=false) must have safety_hold=true.",
    ]

    def vA(r): return r["stock_level"]<=r["reorder_point"] and not r["reorder_pending"]
    def vB(r): return r["is_perishable"] and r["days_until_expiry"]<=expiry_days and r["stock_level"]>overstock_qty and not r["markdown_flagged"]
    def vC(r): return r["selling_price"]<r["cost_price"]*markup and not r["pricing_alert"]
    def vD(r): return r["is_hazardous"] and not r["location_compliant"] and not r["safety_hold"]

    def base(iid):
        cost = 100.0
        rp = 50
        return dict(item_id=iid, stock_level=rp+100, reorder_point=rp, reorder_pending=False,
                    is_perishable=False, days_until_expiry=30, markdown_flagged=False,
                    cost_price=cost, selling_price=round(cost*(markup+0.5),2), pricing_alert=False,
                    is_hazardous=False, location_compliant=True, safety_hold=False)

    rows = [base(f"ITEM-{i+1:03d}") for i in range(15)]

    # TRUE violations
    rp0 = rows[0]["reorder_point"]
    rows[0].update(dict(stock_level=rp0-10, reorder_pending=False))                               # Rule-A easy
    rows[2].update(dict(is_perishable=True, days_until_expiry=expiry_days-2,
                        stock_level=overstock_qty+80, markdown_flagged=False))                     # Rule-B medium
    rows[4].update(dict(selling_price=round(rows[4]["cost_price"]*(markup-0.4),2), pricing_alert=False))  # Rule-C medium
    rows[6].update(dict(is_hazardous=True, location_compliant=False, safety_hold=False))           # Rule-D hard
    rp8 = rows[8]["reorder_point"]
    rows[8].update(dict(stock_level=rp8, reorder_pending=False))  # Rule-A: exactly AT reorder_point (<=)

    # NEAR-MISSES
    # Near-miss Rule-B: perishable + low expiry but stock NOT overstocked
    rows[10].update(dict(is_perishable=True, days_until_expiry=expiry_days-2, stock_level=overstock_qty-30))
    # Near-miss Rule-B: perishable + overstocked but expiry is fine
    rows[11].update(dict(is_perishable=True, days_until_expiry=expiry_days+5, stock_level=overstock_qty+50))
    # Near-miss Rule-D: hazardous but location IS compliant
    rows[12].update(dict(is_hazardous=True, location_compliant=True, safety_hold=False))
    # Near-miss Rule-C: price exactly AT margin floor (not below)
    rows[13].update(dict(selling_price=round(rows[13]["cost_price"]*markup,2), pricing_alert=False))

    gold = []
    for row in rows:
        v = [n for n,f in [("Rule-A",vA),("Rule-B",vB),("Rule-C",vC),("Rule-D",vD)] if f(row)]
        if v:
            gold.append({"row_id":row["item_id"],"rules_violated":v,
                         "difficulty":difficulty_for(v,{"Rule-D"},{"Rule-B","Rule-C"})})

    headers = ["item_id","stock_level","reorder_point","reorder_pending",
               "is_perishable","days_until_expiry","markdown_flagged",
               "cost_price","selling_price","pricing_alert",
               "is_hazardous","location_compliant","safety_hold"]
    csv_rows = [{h:bv(v) if isinstance(v,bool) else v for h,v in row.items()} for row in rows]
    return {"category":"inventory_rules","rules":rules,
            "data_csv":fmt_csv(headers,csv_rows),"gold_violations":gold}


# ---------------------------------------------------------------------------
# Category 4: loan_approval
# ---------------------------------------------------------------------------

def make_loan_approval(seed: int, variant: int) -> dict:
    RNG.seed(seed)
    min_score = [620,600,640,620,580,650,620,600,640,620][variant%10]
    dti_limit = [0.43,0.40,0.45,0.43,0.38,0.43,0.43,0.40,0.45,0.43][variant%10]
    tax_years = [2,2,3,2,2,2,3,2,2,2][variant%10]
    income_thr = [80000,75000,90000,80000,70000,85000,80000,75000,90000,80000][variant%10]
    qual_score = min_score+40
    qual_dti = round(dti_limit-0.07,2)

    rules = [
        f"Rule-A: Applications with credit_score < {min_score} must have status=declined.",
        f"Rule-B: Applications where loan_type=mortgage AND debt_to_income > {dti_limit} must have review_required=true.",
        f"Rule-C: Self-employed applicants (employment_type=self_employed) with tax_returns_filed < {tax_years} must have docs_incomplete=true.",
        f"Rule-D: A 'qualified applicant' is defined as: credit_score >= {qual_score} AND debt_to_income <= {qual_dti} AND income > {income_thr}. Qualified applicants must NOT be sent to manual review (manual_review must be false). Flag qualified applicants where manual_review=true.",
    ]

    def vA(r): return r["credit_score"]<min_score and r["status"]!="declined"
    def vB(r): return r["loan_type"]=="mortgage" and r["debt_to_income"]>dti_limit and not r["review_required"]
    def vC(r): return r["employment_type"]=="self_employed" and r["tax_returns_filed"]<tax_years and not r["docs_incomplete"]
    def qualified(r): return r["credit_score"]>=qual_score and r["debt_to_income"]<=qual_dti and r["income"]>income_thr
    def vD(r): return qualified(r) and r["manual_review"]

    def base(aid):
        return dict(app_id=aid, credit_score=700, debt_to_income=round(dti_limit-0.10,3),
                    loan_type="personal", employment_type="employed",
                    tax_returns_filed=tax_years+1, income=income_thr+30000,
                    status="pending", review_required=True, docs_incomplete=False, manual_review=False)

    rows = [base(f"APP-{i+1:03d}") for i in range(15)]

    # TRUE violations
    rows[0].update(dict(credit_score=min_score-40, status="pending"))                           # Rule-A easy
    rows[2].update(dict(loan_type="mortgage", debt_to_income=round(dti_limit+0.05,3), review_required=False))  # Rule-B medium
    rows[4].update(dict(employment_type="self_employed", tax_returns_filed=tax_years-1, docs_incomplete=False)) # Rule-C medium
    rows[6].update(dict(credit_score=qual_score+30, debt_to_income=round(qual_dti-0.05,3),
                        income=income_thr+25000, manual_review=True))                            # Rule-D hard
    rows[8].update(dict(credit_score=qual_score+50, debt_to_income=round(qual_dti-0.03,3),
                        income=income_thr+15000, manual_review=True))                            # Rule-D hard

    # NEAR-MISSES
    # Near-miss Rule-B: mortgage + high DTI but review_required=true already (compliant)
    rows[10].update(dict(loan_type="mortgage", debt_to_income=round(dti_limit+0.05,3), review_required=True))
    # Near-miss Rule-D: 2 of 3 conditions met (high credit + low DTI but income too low)
    rows[11].update(dict(credit_score=qual_score+20, debt_to_income=round(qual_dti-0.04,3),
                         income=income_thr-5000, manual_review=True))  # NOT qualified because income too low
    # Near-miss Rule-C: self-employed with sufficient tax returns
    rows[12].update(dict(employment_type="self_employed", tax_returns_filed=tax_years, docs_incomplete=False))
    # Near-miss Rule-D: qualified but manual_review=false (correct routing, no violation)
    rows[13].update(dict(credit_score=qual_score+60, debt_to_income=round(qual_dti-0.05,3),
                         income=income_thr+40000, manual_review=False))

    gold = []
    for row in rows:
        v = [n for n,f in [("Rule-A",vA),("Rule-B",vB),("Rule-C",vC),("Rule-D",vD)] if f(row)]
        if v:
            gold.append({"row_id":row["app_id"],"rules_violated":v,
                         "difficulty":difficulty_for(v,{"Rule-D"},{"Rule-B","Rule-C"})})

    headers = ["app_id","credit_score","debt_to_income","loan_type","employment_type",
               "tax_returns_filed","income","status","review_required","docs_incomplete","manual_review"]
    csv_rows = [{h:bv(v) if isinstance(v,bool) else v for h,v in row.items()} for row in rows]
    return {"category":"loan_approval","rules":rules,
            "data_csv":fmt_csv(headers,csv_rows),"gold_violations":gold}


# ---------------------------------------------------------------------------
# Category 5: access_control
# ---------------------------------------------------------------------------

def make_access_control(seed: int, variant: int) -> dict:
    RNG.seed(seed)
    max_login = [90,60,120,90,60,90,120,60,90,90][variant%10]
    max_review = [180,120,180,180,120,180,180,120,180,180][variant%10]
    min_level = [3,2,4,3,3,3,4,2,3,3][variant%10]
    priv_level = min_level+1

    rules = [
        "Rule-A: Users with role=viewer must have edit_permission=false.",
        f"Rule-B: Users with access_level >= {min_level} who are not in the security group (in_security_group=false) must have access_flag=true.",
        f"Rule-C: Admin users (role=admin) must have mfa_enabled=true AND days_since_login <= {max_login}. Non-compliant admins must have compliance_ok=false.",
        f"Rule-D: A 'privileged user' is defined as: role=admin OR role=superuser OR access_level >= {priv_level}. Privileged users with days_since_security_review > {max_review} must have review_overdue=true.",
    ]

    def vA(r): return r["role"]=="viewer" and r["edit_permission"]
    def vB(r): return r["access_level"]>=min_level and not r["in_security_group"] and not r["access_flag"]
    def vC(r): return r["role"]=="admin" and (not r["mfa_enabled"] or r["days_since_login"]>max_login) and r["compliance_ok"]
    def privileged(r): return r["role"] in ("admin","superuser") or r["access_level"]>=priv_level
    def vD(r): return privileged(r) and r["days_since_security_review"]>max_review and not r["review_overdue"]

    def base(uid):
        return dict(user_id=uid, role="editor", access_level=min_level-1,
                    edit_permission=False, in_security_group=True, access_flag=False,
                    mfa_enabled=True, days_since_login=30, compliance_ok=True,
                    days_since_security_review=max_review-30, review_overdue=False)

    rows = [base(f"USR-{i+1:03d}") for i in range(15)]

    # TRUE violations
    rows[0].update(dict(role="viewer", edit_permission=True, access_level=1))                    # Rule-A easy
    rows[2].update(dict(access_level=min_level+1, in_security_group=False, access_flag=False))   # Rule-B medium
    rows[4].update(dict(role="admin", mfa_enabled=False, days_since_login=30, compliance_ok=True))  # Rule-C medium
    rows[6].update(dict(role="editor", access_level=priv_level,
                        days_since_security_review=max_review+40, review_overdue=False,
                        in_security_group=True))                                                   # Rule-D hard (access_level branch)
    rows[8].update(dict(role="superuser", access_level=2,
                        days_since_security_review=max_review+70, review_overdue=False))           # Rule-D hard (role branch)

    # NEAR-MISSES
    # Near-miss Rule-B: high access level but IN security group (compliant)
    rows[10].update(dict(access_level=min_level+1, in_security_group=True, access_flag=False))
    # Near-miss Rule-C: admin with MFA but just over login threshold → actually IS a violation... avoid
    # Near-miss Rule-C: admin + MFA + login within limit (compliant)
    rows[11].update(dict(role="admin", mfa_enabled=True, days_since_login=max_login-10, compliance_ok=True))
    # Near-miss Rule-D: privileged via role=admin but review is recent enough
    rows[12].update(dict(role="admin", mfa_enabled=True, days_since_login=30,
                         days_since_security_review=max_review-10, review_overdue=False, compliance_ok=False))
    # Near-miss Rule-D: access_level = priv_level-1 (just below threshold, NOT privileged)
    rows[13].update(dict(role="editor", access_level=priv_level-1,
                         days_since_security_review=max_review+50, review_overdue=False))

    gold = []
    for row in rows:
        v = [n for n,f in [("Rule-A",vA),("Rule-B",vB),("Rule-C",vC),("Rule-D",vD)] if f(row)]
        if v:
            gold.append({"row_id":row["user_id"],"rules_violated":v,
                         "difficulty":difficulty_for(v,{"Rule-D"},{"Rule-B","Rule-C"})})

    headers = ["user_id","role","access_level","edit_permission","in_security_group",
               "access_flag","mfa_enabled","days_since_login","compliance_ok",
               "days_since_security_review","review_overdue"]
    csv_rows = [{h:bv(v) if isinstance(v,bool) else v for h,v in row.items()} for row in rows]
    return {"category":"access_control","rules":rules,
            "data_csv":fmt_csv(headers,csv_rows),"gold_violations":gold}


# ---------------------------------------------------------------------------

def generate() -> list[dict]:
    datasets = []
    generators = [make_order_compliance, make_hr_policy, make_inventory_rules,
                  make_loan_approval, make_access_control]
    idx = 0
    for gen in generators:
        for variant in range(10):
            entry = gen(1000 + idx*17, variant)
            entry["id"] = idx
            datasets.append(entry)
            idx += 1
    return datasets


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    datasets = generate()

    problems = [f"id={d['id']} {d['category']}: {len(d['gold_violations'])} violations"
                for d in datasets if not (3 <= len(d["gold_violations"]) <= 8)]
    if problems:
        print("WARNING — unexpected violation counts:")
        for p in problems: print(f"  {p}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(datasets, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(datasets)} datasets → {OUTPUT_PATH}")
    by_cat: dict[str, list] = {}
    for d in datasets:
        by_cat.setdefault(d["category"], []).append(len(d["gold_violations"]))
    for cat, counts in sorted(by_cat.items()):
        print(f"  {cat}: avg {sum(counts)/len(counts):.1f} violations/dataset "
              f"(min {min(counts)}, max {max(counts)}) | 15 rows each")

    # Show near-miss structure for one sample
    print("\nSample (order_compliance v0) near-miss check:")
    sample = datasets[0]
    gold_ids = {v["row_id"] for v in sample["gold_violations"]}
    rows = sample["data_csv"].split("\n")
    print(f"  Gold violations: {sorted(gold_ids)}")
    print(f"  Total rows: {len(rows)-1} (+ 1 header)")
