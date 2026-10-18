"""Flatten the relational ETO dataset into one ML training table (one row per RFQ).

Columns are grouped by prefix so leakage is easy to avoid:
  f_   features known at quote time (safe to use for bid/no-bid, win, price, cost and hours models)
  q_   our quote decisions (safe for win / outcome models; they ARE the estimate for cost models)
  y_   targets / outcomes known only later (never use as features for the quote-time models)
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, "csv")
rd = lambda n: pd.read_csv(os.path.join(CSV, f"{n}.csv"))

rfqs, cust, est = rd("rfqs"), rd("customers"), rd("estimators")
quotes, outc, proj = rd("quotes"), rd("quote_outcomes"), rd("projects")
reqs, clauses, bom = rd("rfq_requirements"), rd("contract_clauses"), rd("project_bom_actuals")
comm, cap, sq = rd("commodity_prices"), rd("shop_capacity"), rd("supplier_quotes")

df = rfqs.merge(cust[["customer_id", "segment", "credit_rating", "relationship_since", "procurement_style", "payment_behaviour_days"]],
                on="customer_id", how="left")
df["received_date"] = pd.to_datetime(df.received_date)
df["bid_due_date"] = pd.to_datetime(df.bid_due_date)

# ---------------- f_: RFQ / customer features known at quote time
f = pd.DataFrame({"rfq_id": df.rfq_id, "customer_id": df.customer_id, "received_date": df.received_date.dt.date})
f["f_year"] = df.received_date.dt.year
f["f_month"] = df.received_date.dt.month
f["f_days_to_bid_due"] = (df.bid_due_date - df.received_date).dt.days
for c in ["family_id", "channel", "tender_type", "end_user_industry", "site_region", "quantity", "capacity_value", "installed_power_kw",
          "estimated_weight_t", "primary_material", "design_code", "operating_environment", "complexity_score_1to5", "customisation_pct",
          "scope_includes_installation", "scope_includes_civil", "required_delivery_weeks", "ld_clause_pct_per_week", "ld_cap_pct",
          "warranty_months", "performance_guarantee", "payment_terms", "inco_terms", "currency", "rfq_document_pages",
          "num_drawings_attached", "competitors_invited", "spec_clarity_1to5"]:
    f[f"f_{c}"] = df[c].values
f["f_customer_segment"] = df.segment
f["f_customer_credit_rating"] = df.credit_rating
f["f_customer_procurement_style"] = df.procurement_style
f["f_customer_payment_days"] = df.payment_behaviour_days
f["f_customer_tenure_years"] = ((df.received_date - pd.to_datetime(df.relationship_since)).dt.days / 365.25).clip(lower=0).round(1)

# customer history strictly before this RFQ (leakage-safe)
hist = df[["rfq_id", "customer_id", "received_date"]].merge(outc[["rfq_id", "outcome", "decision_date"]], on="rfq_id")
hist["decision_date"] = pd.to_datetime(hist.decision_date)
hist = hist.sort_values("received_date")
prior_rfqs, prior_wins, prior_decided = [], [], []
by_cust = {c: g for c, g in hist.groupby("customer_id")}
for r in hist.itertuples():
    g = by_cust[r.customer_id]
    earlier = g[g.received_date < r.received_date]
    known = earlier[earlier.decision_date < r.received_date]
    prior_rfqs.append(len(earlier))
    prior_decided.append(int(known.outcome.isin(["Won", "Lost"]).sum()))
    prior_wins.append(int((known.outcome == "Won").sum()))
h = pd.DataFrame({"rfq_id": hist.rfq_id, "f_customer_prior_rfqs": prior_rfqs, "f_customer_prior_decided_bids": prior_decided,
                  "f_customer_prior_wins": prior_wins})
f = f.merge(h, on="rfq_id", how="left")

rq = reqs.groupby("rfq_id").agg(f_n_requirements=("requirement_id", "size"),
                                f_n_mandatory_requirements=("mandatory", lambda s: int((s == "Y").sum())))
f = f.merge(rq, on="rfq_id", how="left")
cl = clauses.groupby("rfq_id").agg(f_n_high_risk_clauses=("risk_level", lambda s: int((s == "High").sum())),
                                   f_n_medium_risk_clauses=("risk_level", lambda s: int((s == "Medium").sum())))
f = f.merge(cl, on="rfq_id", how="left")

steel = comm[comm.commodity == "Steel HR coil"].copy()
steel["month"] = pd.to_datetime(steel.month)
steel = steel.set_index("month").price
f["f_steel_price_at_rfq"] = [round(float(steel.asof(pd.Timestamp(d))), 0) for d in df.received_date]

fab = cap[cap.work_center_id == "WC-FAB"].copy()
fab["week_start"] = pd.to_datetime(fab.week_start)
fab = fab.set_index("week_start").sort_index().utilisation_pct
f["f_fab_load_next_12w_pct"] = [round(float(fab[(fab.index >= d) & (fab.index < d + pd.Timedelta(weeks=12))].mean()), 1) for d in df.received_date]

# ---------------- q_: our quote (bids only)
qcols = ["quote_id", "rfq_id", "quote_date", "estimator_id", "revision", "pricing_model", "est_material_cost_inr", "est_engineering_hrs",
         "est_fabrication_hrs", "est_assembly_hrs", "est_testing_hrs", "est_labour_cost_inr", "est_subcontract_cost_inr", "contingency_pct",
         "est_total_cost_inr", "target_margin_pct", "quoted_price_inr", "quoted_lead_time_weeks", "validity_days", "quote_preparation_hrs",
         "num_deviations_taken", "similar_past_quote_ref", "approval_level"]
q = quotes[qcols].merge(est[["estimator_id", "experience_years"]], on="estimator_id", how="left").rename(columns={"experience_years": "estimator_experience_years"})
q = q.rename(columns={c: f"q_{c}" for c in q.columns if c not in ("quote_id", "rfq_id")})
f = f.merge(q, on="rfq_id", how="left")
f["q_lead_time_gap_weeks"] = f.q_quoted_lead_time_weeks - f.f_required_delivery_weeks
f["q_price_per_tonne_inr"] = (f.q_quoted_price_inr / (f.f_estimated_weight_t * f.f_quantity)).round(0)
nsq = sq.groupby("quote_id").size().rename("q_n_supplier_quotes")
f = f.merge(nsq, left_on="quote_id", right_index=True, how="left")

# ---------------- y_: outcomes and actuals
o = outc[["rfq_id", "bid_decision", "no_bid_reason", "outcome", "decision_date", "loss_reason", "winning_competitor_id",
          "winning_price_inr_if_known", "final_order_value_inr"]].copy()
o = o.rename(columns={c: f"y_{c}" for c in o.columns if c != "rfq_id"})
f = f.merge(o, on="rfq_id", how="left")
f["y_is_bid"] = (f.y_bid_decision == "Bid").astype(int)
f["y_is_won"] = np.where(f.y_outcome == "Won", 1, np.where(f.y_outcome == "Lost", 0, np.nan))
f = f.merge(quotes[["quote_id", "discount_pct"]].rename(columns={"discount_pct": "y_discount_pct"}), on="quote_id", how="left")
f.loc[f.y_outcome != "Won", "y_discount_pct"] = np.nan

p = proj.copy()
p["y_delivery_slip_weeks"] = ((pd.to_datetime(p.actual_delivery_date) - pd.to_datetime(p.planned_delivery_date)).dt.days / 7).round(1)
added = bom[bom.added_after_quote == "Y"].groupby("project_id").size().rename("y_n_items_added_after_quote")
p = p.merge(added, left_on="project_id", right_index=True, how="left")
p.loc[p.project_status == "Closed", "y_n_items_added_after_quote"] = p.y_n_items_added_after_quote.fillna(0)
pc = ["project_id", "quote_id", "project_status", "order_date", "change_order_value_inr", "actual_material_cost_inr", "actual_engineering_hrs",
      "actual_fabrication_hrs", "actual_assembly_hrs", "actual_testing_hrs", "actual_labour_cost_inr", "actual_subcontract_cost_inr",
      "ld_paid_inr", "warranty_cost_inr", "actual_total_cost_inr", "actual_margin_pct", "num_engineering_changes", "root_cause_of_overrun"]
p = p[pc + ["y_delivery_slip_weeks", "y_n_items_added_after_quote"]]
p = p.rename(columns={c: f"y_{c}" for c in pc if c not in ("project_id", "quote_id")})
f = f.merge(p, on="quote_id", how="left")
closed = f.y_project_status == "Closed"
f["y_cost_overrun_ratio"] = (f.y_actual_total_cost_inr / f.q_est_total_cost_inr).round(4)
f["y_is_cost_overrun_gt5pct"] = np.where(closed, (f.y_cost_overrun_ratio > 1.05).astype(float), np.nan)
f["y_ld_paid_flag"] = np.where(closed, (f.y_ld_paid_inr > 0).astype(float), np.nan)
f["y_margin_erosion_pts"] = np.where(closed, (f.q_target_margin_pct - f.y_actual_margin_pct).round(1), np.nan)

# time-based split on RFQ received date
f["split"] = np.select([f.received_date.astype(str) < "2025-01-01", f.received_date.astype(str) < "2025-07-01"], ["train", "valid"], "test")

cols = ["rfq_id", "quote_id", "project_id", "customer_id", "received_date", "split"] + \
       [c for c in f.columns if c.startswith("f_")] + [c for c in f.columns if c.startswith("q_")] + [c for c in f.columns if c.startswith("y_")]
f = f[cols].sort_values("received_date")
f.to_csv(os.path.join(CSV, "ml_training_table.csv"), index=False)
print("ml_training_table", f.shape, f.split.value_counts().to_dict())
