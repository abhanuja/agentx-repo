"""Quick baseline models to confirm each ML use case has learnable signal in the sample data.
Writes csv/../ml_baseline_results.csv. Gradient boosting on the flat table; not tuned."""
import os, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGC, HistGradientBoostingRegressor as HGR
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, "csv")
df = pd.read_csv(os.path.join(CSV, "ml_training_table.csv"))

F = [c for c in df.columns if c.startswith("f_")]
Q = [c for c in df.columns if c.startswith("q_") and c not in ("q_quote_date", "q_similar_past_quote_ref", "q_estimator_id")]
for c in F + Q:
    if not pd.api.types.is_numeric_dtype(df[c]):
        df[c] = df[c].astype("category")

res = []


def mape(y, p):
    return float(np.mean(np.abs(p - y) / np.abs(y)))


def add(uc, task, target, n, metric, model, baseline, note=""):
    res.append(dict(use_case=uc, task=task, target=target, rows_used=n, metric=metric, model_score=round(model, 3),
                    naive_baseline=round(baseline, 3), note=note))


def time_split(d):
    return d[d.split != "test"], d[d.split == "test"]


def rand_split(d):
    return train_test_split(d, test_size=0.25, random_state=7)


def clf(d, feats, y, split=time_split):
    tr, te = split(d)
    m = HGC(categorical_features="from_dtype", max_iter=300, learning_rate=0.05, random_state=0).fit(tr[feats], tr[y])
    return te, m


def reg(d, feats, y, split=rand_split):
    tr, te = split(d)
    m = HGR(categorical_features="from_dtype", max_iter=400, learning_rate=0.05, random_state=0).fit(tr[feats], tr[y])
    return te, m.predict(te[feats])


# 1 bid / no-bid
te, m = clf(df, F, "y_is_bid")
add("Bid / no-bid scoring", "Binary classification", "y_is_bid", len(df), "ROC AUC (time split)", roc_auc_score(te.y_is_bid, m.predict_proba(te[F])[:, 1]), 0.5)

# 2 win probability
d = df[df.y_is_won.notna()]
te, m = clf(d, F + Q, "y_is_won")
add("Win probability", "Binary classification", "y_is_won", len(d), "ROC AUC (time split)", roc_auc_score(te.y_is_won, m.predict_proba(te[F + Q])[:, 1]), 0.5,
    "Price vs cost, lead-time gap, segment and number of bidders drive the outcome")

# 3 loss reason
d = df[df.y_outcome == "Lost"].copy()
d["y_loss_reason"] = d.y_loss_reason.astype(str)
te, m = clf(d, F + Q, "y_loss_reason")
pred = m.predict(te[F + Q])
add("Loss-reason prediction", "Multi-class classification", "y_loss_reason", len(d), "Accuracy (time split)", accuracy_score(te.y_loss_reason, pred),
    (te.y_loss_reason == d.y_loss_reason.mode()[0]).mean())

# 4 discount
d = df[df.y_discount_pct.notna()]
te, p = reg(d, F + Q, "y_discount_pct")
add("Negotiation discount", "Regression", "y_discount_pct", len(d), "MAE (pct points)", mean_absolute_error(te.y_discount_pct, p),
    mean_absolute_error(te.y_discount_pct, np.full(len(te), d.y_discount_pct.mean())), "Weak signal by design (random in the generator)")

closed = df[df.y_project_status == "Closed"].copy()
# 5 cost estimate correction
te, p = reg(closed, F + Q, "y_cost_overrun_ratio")
corrected = te.q_est_total_cost_inr * p
add("Cost estimate correction (should-cost)", "Regression", "y_actual_total_cost_inr", len(closed), "MAPE of total cost",
    mape(te.y_actual_total_cost_inr, corrected), mape(te.y_actual_total_cost_inr, te.q_est_total_cost_inr),
    "Baseline = estimator's own estimate; model learns the estimating bias")
# 5b cost from specs only
closed["log_cost"] = np.log(closed.y_actual_total_cost_inr)
te, p = reg(closed, F, "log_cost")
add("Cost prediction from specs only (no estimate)", "Regression", "y_actual_total_cost_inr", len(closed), "MAPE of total cost",
    mape(te.y_actual_total_cost_inr, np.exp(p)), mape(te.y_actual_total_cost_inr, np.full(len(te), closed.y_actual_total_cost_inr.median())),
    "Useful for instant budgetary quotes before detailed costing")

# 6 hours
for y, e in [("y_actual_engineering_hrs", "q_est_engineering_hrs"), ("y_actual_fabrication_hrs", "q_est_fabrication_hrs"),
             ("y_actual_assembly_hrs", "q_est_assembly_hrs")]:
    closed["_t"] = np.log(closed[y])
    closed["_r"] = np.log(closed[y] / closed[e])
    te, p = reg(closed, F + ["q_estimator_experience_years", e], "_r")
    add(f"Hours estimation – {y.split('_')[2]}", "Regression", y, len(closed), "MAPE", mape(te[y], te[e] * np.exp(p)), mape(te[y], te[e]),
        "Baseline = estimator's hours; model corrects them using specs and estimator experience")

# 7 overrun risk, LD, slip
te, m = clf(closed, F + Q, "y_is_cost_overrun_gt5pct", split=rand_split)
add("Cost-overrun risk (>5%)", "Binary classification", "y_is_cost_overrun_gt5pct", len(closed), "ROC AUC", roc_auc_score(te.y_is_cost_overrun_gt5pct, m.predict_proba(te[F + Q])[:, 1]), 0.5)
te, p = reg(closed, F + Q, "y_delivery_slip_weeks")
add("Delivery-delay prediction", "Regression", "y_delivery_slip_weeks", len(closed), "MAE (weeks)", mean_absolute_error(te.y_delivery_slip_weeks, p),
    mean_absolute_error(te.y_delivery_slip_weeks, np.full(len(te), closed.y_delivery_slip_weeks.mean())), "Shop load, complexity and spec clarity drive slip")
te, m = clf(closed, F + Q, "y_ld_paid_flag", split=rand_split)
add("LD (penalty) exposure", "Binary classification", "y_ld_paid_flag", len(closed), "ROC AUC", roc_auc_score(te.y_ld_paid_flag, m.predict_proba(te[F + Q])[:, 1]), 0.5)
te, p = reg(closed, F + Q, "y_margin_erosion_pts")
add("Margin-erosion prediction", "Regression", "y_margin_erosion_pts", len(closed), "MAE (pct points)", mean_absolute_error(te.y_margin_erosion_pts, p),
    mean_absolute_error(te.y_margin_erosion_pts, np.full(len(te), closed.y_margin_erosion_pts.mean())))
te, p = reg(closed, F, "y_num_engineering_changes")
add("Engineering-change count", "Regression (count)", "y_num_engineering_changes", len(closed), "MAE", mean_absolute_error(te.y_num_engineering_changes, p),
    mean_absolute_error(te.y_num_engineering_changes, np.full(len(te), closed.y_num_engineering_changes.mean())))
closed["_added"] = (closed.y_n_items_added_after_quote > 0).astype(int)
te, m = clf(closed, F, "_added", split=rand_split)
add("Missed-BOM-item risk", "Binary classification", "y_n_items_added_after_quote > 0", len(closed), "ROC AUC", roc_auc_score(te._added, m.predict_proba(te[F])[:, 1]), 0.5)

# 8 quote effort
d = df[df.q_quote_preparation_hrs.notna()]
te, p = reg(d, F, "q_quote_preparation_hrs")
add("Quote-effort prediction", "Regression", "q_quote_preparation_hrs", len(d), "MAPE", mape(te.q_quote_preparation_hrs, p),
    mape(te.q_quote_preparation_hrs, np.full(len(te), d.q_quote_preparation_hrs.mean())), "For bid resource planning")

# 9 supplier price & lead time
sq = pd.read_csv(os.path.join(CSV, "supplier_quotes.csv")).merge(pd.read_csv(os.path.join(CSV, "item_master.csv")), on="item_id")
sq = sq.merge(pd.read_csv(os.path.join(CSV, "suppliers.csv")), on="supplier_id", suffixes=("", "_sup"))
sq["month"] = pd.to_datetime(sq.quote_received_date).dt.month
sq["year"] = pd.to_datetime(sq.quote_received_date).dt.year
sf = ["item_category", "uom", "commodity_driver", "standard_cost_inr", "standard_lead_time_days", "qty", "country", "avg_lead_time_days",
      "on_time_delivery_pct", "quality_rating", "year", "month"]
for c in sf:
    if not pd.api.types.is_numeric_dtype(sq[c]):
        sq[c] = sq[c].astype("category")
sq["log_p"] = np.log(sq.quoted_unit_price_inr)
te, p = reg(sq, sf, "log_p")
add("Bought-out price prediction", "Regression", "supplier_quotes.quoted_unit_price_inr", len(sq), "MAPE", mape(te.quoted_unit_price_inr, np.exp(p)),
    mape(te.quoted_unit_price_inr, te.standard_cost_inr), "Baseline = item standard cost")
te, p = reg(sq, sf, "quoted_lead_time_days")
add("Supplier lead-time prediction", "Regression", "supplier_quotes.quoted_lead_time_days", len(sq), "MAE (days)", mean_absolute_error(te.quoted_lead_time_days, p),
    mean_absolute_error(te.quoted_lead_time_days, te.standard_lead_time_days), "Baseline = item standard lead time")

# 10 NLP on requirements
rq = pd.read_csv(os.path.join(CSV, "rfq_requirements.csv"))
tr, te = train_test_split(rq, test_size=0.25, random_state=7)
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit(tr.requirement_text)
lr = LogisticRegression(max_iter=2000).fit(vec.transform(tr.requirement_text), tr.category)
add("Requirement categorisation (NLP)", "Text classification", "rfq_requirements.category", len(rq), "Macro F1",
    f1_score(te.category, lr.predict(vec.transform(te.requirement_text)), average="macro"), 1 / rq.category.nunique(),
    "Near-perfect because text is templated; use for pipeline testing, not model selection")
rq2 = rq.merge(df[["rfq_id", "f_spec_clarity_1to5", "f_ld_cap_pct", "f_warranty_months", "f_customisation_pct", "f_required_delivery_weeks",
                    "f_scope_includes_installation"]], on="rfq_id")
rq2["ctx"] = rq2.category + " | clarity" + rq2.f_spec_clarity_1to5.astype(str) + " | ldcap" + rq2.f_ld_cap_pct.astype(str) \
    + " | warr" + rq2.f_warranty_months.astype(str) + " | cust" + (rq2.f_customisation_pct > 70).astype(str) \
    + " | inst" + rq2.f_scope_includes_installation.astype(str) + " | long" + (rq2.f_required_delivery_weeks > 24).astype(str)
rq2["doc"] = rq2.requirement_text + " " + rq2.ctx
tr, te = train_test_split(rq2, test_size=0.25, random_state=7)
vec2 = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit(tr.doc)
lr2 = LogisticRegression(max_iter=3000, class_weight="balanced").fit(vec2.transform(tr.doc), tr.compliance_status)
maj = np.full(len(te), rq2.compliance_status.mode()[0])
add("Compliance / deviation suggestion", "Text + context classification", "rfq_requirements.compliance_status", len(rq2), "Macro F1",
    f1_score(te.compliance_status, lr2.predict(vec2.transform(te.doc)), average="macro"), f1_score(te.compliance_status, maj, average="macro"),
    "Driven by clause type and RFQ context (LD cap, warranty, clarity, scope); real data needs your past compliance matrices")

out = pd.DataFrame(res)
out.to_csv(os.path.join(BASE, "ml_baseline_results.csv"), index=False)
print(out[["use_case", "metric", "model_score", "naive_baseline"]].to_string())
