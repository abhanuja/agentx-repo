"""
Synthetic data generator for an Engineer-to-Order (ETO) AI quoting program.

Generic ETO manufacturer: custom-engineered equipment built per customer spec.
Produces linked tables covering RFQ intake -> quoting -> win/loss -> project actuals,
plus master data (customers, items, suppliers, competitors, rates, commodities, capacity).

Relationships baked in so models can actually learn something:
  * cost and hours scale with capacity, power, weight and complexity
  * estimators under-estimate high-complexity / high-customisation jobs
  * win probability falls as our price rises above the competitor's and lead time grows
  * actual material cost tracks commodity price movement between quote and order
  * LD / penalty exposure rises when actual delivery slips past plan

Run:  python generate_eto_data.py  -> writes CSVs into ./csv
"""
import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
os.makedirs(OUT, exist_ok=True)

START, END = pd.Timestamp("2021-01-01"), pd.Timestamp("2026-09-30")
N_CUST = int(os.environ.get("ETO_N_CUST", 600))
N_SUP = int(os.environ.get("ETO_N_SUP", 150))
N_RFQ = int(os.environ.get("ETO_N_RFQ", 6000))

# RFQs that have a matching customer PDF (see build_rfq_pdfs.py). Their parameters are pinned so the
# generated quote, outcome and project rows stay consistent with the documents.
PINNED_CUSTOMERS = {
    "C0033": dict(customer_name="Meridian Minerals Pvt Ltd", industry="Mining", region="India-West", country="India", segment="Standard", credit_rating="A"),
    "C0110": dict(customer_name="Nova Logistics Group", industry="Logistics", region="Africa", country="Kenya", segment="Key", credit_rating="B"),
    "C0137": dict(customer_name="Delta Wassertechnik GmbH", industry="Water & Wastewater", region="Europe", country="Germany", segment="Key", credit_rating="B"),
}
PINNED_RFQS = {
    "RFQ-2023-0105": dict(customer_id="C0033", received_date="2023-08-05", bid_due_date="2023-08-23", family_id="PF02",
                          rfq_title="Reagent Dosing & Transfer Skid - 61.1 m3/h", channel="Customer portal", tender_type="Firm",
                          quantity=1, capacity_value=61.1, installed_power_kw=165.7, estimated_weight_t=30.4, primary_material="SS316L",
                          design_code="ASME VIII Div 1", operating_environment="Indoor dusty", complexity_score_1to5=4, customisation_pct=81.0,
                          scope_includes_installation="Y", scope_includes_civil="N", required_delivery_weeks=16, ld_clause_pct_per_week=1.0,
                          ld_cap_pct=5, warranty_months=24, performance_guarantee="N", payment_terms="10/80/10", inco_terms="FOR site",
                          currency="INR", num_drawings_attached=11, competitors_invited=3, spec_clarity_1to5=4),
    "RFQ-2023-0094": dict(customer_id="C0110", received_date="2023-07-21", bid_due_date="2023-08-07", family_id="PF06",
                          rfq_title="Rail-Mounted Goliath Cranes 82.4 t SWL (2 nos)", channel="Tender portal (GeM/eProc)", tender_type="Firm",
                          quantity=2, capacity_value=82.4, installed_power_kw=326.4, estimated_weight_t=85.9, primary_material="S355 / IS 2062 E350",
                          design_code="FEM 1.001", operating_environment="Coastal / saline", complexity_score_1to5=5, customisation_pct=94.0,
                          scope_includes_installation="Y", scope_includes_civil="N", required_delivery_weeks=32, ld_clause_pct_per_week=1.0,
                          ld_cap_pct=20, warranty_months=12, performance_guarantee="N", payment_terms="Net 90 after delivery", inco_terms="CIF",
                          currency="USD", num_drawings_attached=16, competitors_invited=4, spec_clarity_1to5=2),
    "RFQ-2026-0563": dict(customer_id="C0137", received_date="2026-02-15", bid_due_date="2026-03-15", family_id="PF04",
                          rfq_title="End-Cap Welding & Leak-Test Machines - 663 units/h (2 nos)", channel="Sales rep", tender_type="Firm",
                          quantity=2, capacity_value=663.2, installed_power_kw=447.9, estimated_weight_t=56.2, primary_material="SS316L",
                          design_code="CE Machinery Directive", operating_environment="Corrosive", complexity_score_1to5=5, customisation_pct=83.0,
                          scope_includes_installation="Y", scope_includes_civil="N", required_delivery_weeks=21, ld_clause_pct_per_week=2.0,
                          ld_cap_pct=20, warranty_months=18, performance_guarantee="Y", payment_terms="Milestone-based", inco_terms="FOB",
                          currency="EUR", num_drawings_attached=15, competitors_invited=5, spec_clarity_1to5=4),
}
FORCED_OUTCOME = {"RFQ-2023-0105": "Won", "RFQ-2023-0094": "Lost", "RFQ-2026-0563": "Won"}


def pick(seq, n=None, p=None):
    return RNG.choice(seq, size=n, p=p)


def rand_date(start, end, n):
    span = (end - start).days
    return start + pd.to_timedelta(RNG.integers(0, span, n), unit="D")


# ----------------------------------------------------------------- reference data
PRODUCT_FAMILIES = pd.DataFrame([
    ("PF01", "Material Handling System", "Custom conveyors, transfer systems and handling lines", "tph", 5, 400),
    ("PF02", "Process Skid", "Pre-assembled process modules: pumps, valves, instrumentation on a frame", "m3/h", 2, 250),
    ("PF03", "Pressure Vessel / Tank", "Code-stamped vessels, columns and storage tanks", "m3", 1, 150),
    ("PF04", "Special Purpose Machine", "Bespoke automated machines for a specific production step", "units/h", 20, 1200),
    ("PF05", "Power & Control Package", "Custom panels, MCC, power distribution and control systems", "kVA", 50, 3000),
    ("PF06", "Lifting Equipment", "Cranes, hoists, gantries engineered to site", "t", 2, 120),
], columns=["family_id", "family_name", "description", "capacity_unit", "capacity_min", "capacity_max"])

INDUSTRIES = ["Cement", "Steel", "Food & Beverage", "Pharma", "Chemicals", "Oil & Gas",
              "Power", "Automotive", "Mining", "Water & Wastewater", "Paper", "Logistics"]
REGIONS = {"India-South": "India", "India-West": "India", "India-North": "India", "India-East": "India",
           "Middle East": "UAE", "South-East Asia": "Singapore", "Africa": "Kenya", "Europe": "Germany"}
DESIGN_CODES = ["ASME VIII Div 1", "IS 2825", "EN 13445", "IEC 61439", "IS 3177", "FEM 1.001",
                "ISO 13849", "CE Machinery Directive", "API 650", "Customer standard"]
MATERIALS = ["IS 2062 E250", "SS304", "SS316L", "SA516 Gr70", "Duplex 2205", "Aluminium 6061", "MS + Epoxy coat"]
ENVIRONMENTS = ["Indoor clean", "Indoor dusty", "Outdoor", "Corrosive", "Hazardous area Zone 1",
                "Hazardous area Zone 2", "Coastal / saline", "High temperature"]
COMMODITIES = {"Steel HR coil": ("INR/t", 58000), "Stainless steel 304": ("INR/t", 215000),
               "Copper": ("INR/t", 760000), "Aluminium": ("INR/t", 230000),
               "Electronics index": ("index", 100), "Diesel": ("INR/l", 90)}

# ----------------------------------------------------------------- customers
cust_regions = pick(list(REGIONS), N_CUST, p=[.22, .2, .14, .1, .14, .08, .06, .06])
customers = pd.DataFrame({
    "customer_id": [f"C{i:04d}" for i in range(1, N_CUST + 1)],
    "customer_name": [f"{pick(['Apex','Vertex','Sundar','Shakti','Orion','Kaveri','Delta','Nova','Ganga','Atlas','Zenith','Meridian','Prime','Sahyadri','Indus'])} "
                      f"{pick(['Industries','Cements','Steels','Foods','Pharma','Chemicals','Energy','Motors','Minerals','Utilities','Papers','Logistics'])} "
                      f"{pick(['Ltd','Pvt Ltd','LLC','PLC','Group'])}" for _ in range(N_CUST)],
    "industry": pick(INDUSTRIES, N_CUST),
    "region": cust_regions,
    "country": [REGIONS[r] for r in cust_regions],
    "segment": pick(["Strategic", "Key", "Standard", "New"], N_CUST, p=[.1, .25, .45, .2]),
    "credit_rating": pick(["A", "B", "C", "D"], N_CUST, p=[.35, .4, .2, .05]),
    "relationship_since": rand_date(pd.Timestamp("2008-01-01"), pd.Timestamp("2026-06-30"), N_CUST).date,
    "procurement_style": pick(["Direct negotiation", "Open tender", "Limited tender", "EPC contractor"], N_CUST, p=[.35, .25, .25, .15]),
    "payment_behaviour_days": RNG.integers(25, 120, N_CUST),
})
customers["customer_name"] = [f"{n} ({i[-4:]})" if d else n for n, i, d in
                              zip(customers.customer_name, customers.customer_id, customers.customer_name.duplicated(keep=False))]
for cid, fix in PINNED_CUSTOMERS.items():
    for k, v in fix.items():
        customers.loc[customers.customer_id == cid, k] = v

# ----------------------------------------------------------------- suppliers
SUP_CATS = ["Steel & plate", "Stainless & alloys", "Motors & drives", "Gearboxes", "Bearings & transmission",
            "Pumps & valves", "Instrumentation", "Electrical & cables", "PLC & automation", "Fabrication subcontract",
            "Machining subcontract", "Painting & coating", "Fasteners & consumables", "Castings & forgings"]
suppliers = pd.DataFrame({
    "supplier_id": [f"S{i:03d}" for i in range(1, N_SUP + 1)],
    "supplier_name": [f"{pick(['Bharat','Precision','Unique','Global','Metro','Star','Allied','Sri Venkatesh','Techno','United'])} "
                      f"{pick(['Engineering','Traders','Components','Systems','Metals','Automation','Works'])} {i}" for i in range(1, N_SUP + 1)],
    "category": [SUP_CATS[i % len(SUP_CATS)] for i in range(N_SUP)],
    "country": pick(["India", "India", "India", "Germany", "China", "Japan", "Italy", "USA"], N_SUP),
    "avg_lead_time_days": RNG.integers(7, 120, N_SUP),
    "on_time_delivery_pct": np.round(RNG.uniform(62, 99, N_SUP), 1),
    "quality_rating": np.round(RNG.uniform(2.5, 5, N_SUP), 1),
    "approved_vendor": pick(["Y", "Y", "Y", "N"], N_SUP),
})

# ----------------------------------------------------------------- item master
ITEM_TEMPLATES = [
    # category, description stem, uom, make_buy, cost range, supplier category, commodity link
    ("Structural", "Fabricated base frame", "kg", "Make", (180, 260), "Steel & plate", "Steel HR coil"),
    ("Structural", "Support structure & platforms", "kg", "Make", (170, 240), "Steel & plate", "Steel HR coil"),
    ("Structural", "SS wetted parts", "kg", "Make", (520, 780), "Stainless & alloys", "Stainless steel 304"),
    ("Plate", "Shell plate", "kg", "Buy", (70, 95), "Steel & plate", "Steel HR coil"),
    ("Plate", "SS plate", "kg", "Buy", (240, 320), "Stainless & alloys", "Stainless steel 304"),
    ("Drive", "Induction motor", "nos", "Buy", (18000, 420000), "Motors & drives", "Copper"),
    ("Drive", "Variable frequency drive", "nos", "Buy", (25000, 650000), "Motors & drives", "Electronics index"),
    ("Drive", "Helical gearbox", "nos", "Buy", (35000, 900000), "Gearboxes", "Steel HR coil"),
    ("Mechanical", "Bearing housing assembly", "nos", "Buy", (3500, 60000), "Bearings & transmission", "Steel HR coil"),
    ("Mechanical", "Coupling", "nos", "Buy", (4000, 85000), "Bearings & transmission", "Steel HR coil"),
    ("Mechanical", "Machined shaft", "nos", "Make", (6000, 140000), "Machining subcontract", "Steel HR coil"),
    ("Mechanical", "Casting", "kg", "Buy", (160, 420), "Castings & forgings", "Steel HR coil"),
    ("Fluid", "Centrifugal pump", "nos", "Buy", (45000, 1200000), "Pumps & valves", "Stainless steel 304"),
    ("Fluid", "Control valve", "nos", "Buy", (30000, 450000), "Pumps & valves", "Stainless steel 304"),
    ("Fluid", "Piping & fittings", "m", "Make", (900, 7500), "Stainless & alloys", "Stainless steel 304"),
    ("Instrumentation", "Pressure / level transmitter", "nos", "Buy", (18000, 120000), "Instrumentation", "Electronics index"),
    ("Instrumentation", "Flow meter", "nos", "Buy", (60000, 650000), "Instrumentation", "Electronics index"),
    ("Electrical", "Power cable", "m", "Buy", (180, 4200), "Electrical & cables", "Copper"),
    ("Electrical", "Control panel enclosure", "nos", "Make", (40000, 450000), "Electrical & cables", "Steel HR coil"),
    ("Electrical", "Switchgear / breaker", "nos", "Buy", (12000, 900000), "Electrical & cables", "Copper"),
    ("Automation", "PLC & IO", "nos", "Buy", (90000, 1800000), "PLC & automation", "Electronics index"),
    ("Automation", "HMI / SCADA licence", "nos", "Buy", (60000, 700000), "PLC & automation", "Electronics index"),
    ("Finish", "Surface prep & painting", "m2", "Buy", (350, 1400), "Painting & coating", "Diesel"),
    ("Consumable", "Fasteners & consumables kit", "lot", "Buy", (5000, 90000), "Fasteners & consumables", "Steel HR coil"),
    ("Subcontract", "Heavy fabrication subcontract", "kg", "Buy", (90, 160), "Fabrication subcontract", "Steel HR coil"),
]
items = []
iid = 1
for cat, stem, uom, mb, (lo, hi), scat, comm in ITEM_TEMPLATES:
    sizes = ["S", "M", "L", "XL"] if uom == "nos" else ["Std"]
    for s in sizes:
        frac = {"S": .05, "M": .2, "L": .5, "XL": 1.0, "Std": RNG.uniform(.2, .8)}[s]
        cost = lo + (hi - lo) * frac * RNG.uniform(.8, 1.1)
        sup = suppliers[suppliers.category == scat].supplier_id.values
        items.append({
            "item_id": f"IT{iid:04d}",
            "item_description": f"{stem}{'' if s == 'Std' else ' - size ' + s}",
            "item_category": cat,
            "uom": uom,
            "make_buy": mb,
            "material": pick(MATERIALS) if cat in ("Structural", "Plate", "Fluid", "Mechanical") else "",
            "standard_cost_inr": round(cost, 2),
            "standard_lead_time_days": int(RNG.integers(5, 140)),
            "preferred_supplier_id": pick(sup) if len(sup) else "",
            "commodity_driver": comm,
            "active": "Y",
        })
        iid += 1
items = pd.DataFrame(items)

# ----------------------------------------------------------------- competitors, estimators, labour rates
competitors = pd.DataFrame({
    "competitor_id": [f"K{i:02d}" for i in range(1, 9)],
    "competitor_name": ["Thermax-like Corp", "Global EngTech", "Precision Systems Intl", "Low-Cost Fab Co",
                        "EuroProcess GmbH", "Regional Machines", "AsiaPac Equipment", "Integrated Solutions Ltd"],
    "positioning": ["Premium", "Premium", "Mid", "Low-cost", "Premium", "Low-cost", "Mid", "Mid"],
    "strong_families": ["PF02;PF03", "PF04;PF05", "PF01;PF04", "PF01;PF03;PF06", "PF02;PF03", "PF06;PF01", "PF05;PF04", "PF01;PF02;PF05"],
    "typical_price_index": [1.12, 1.08, 1.0, 0.86, 1.15, 0.9, 0.95, 1.0],
})

N_EST = 30
est_exp = np.clip(np.round(RNG.gamma(2.2, 5.0, N_EST)), 1, 32).astype(int)
estimators = pd.DataFrame({
    "estimator_id": [f"E{i:02d}" for i in range(1, N_EST + 1)],
    "estimator_name": [f"Estimator {i:02d}" for i in range(1, N_EST + 1)],
    "discipline": pick(["Mechanical", "Process", "Electrical", "Automation", "Structural"], N_EST, p=[.35, .2, .2, .15, .1]),
    "experience_years": est_exp,
    "location": pick(["Bengaluru", "Pune", "Chennai"], N_EST),
})
# estimator bias: less experience -> more under-estimation
estimators["_bias"] = np.clip(0.97 - 0.004 * (25 - estimators.experience_years), 0.86, 1.0)

labour_rates = []
for yr, infl in zip(range(2020, 2027), [0.88, 0.92, 0.96, 1.0, 1.06, 1.12, 1.19]):
    for res, base in [("Design engineering", 1450), ("Detail drafting", 900), ("Electrical / automation engg", 1600),
                      ("Fabrication", 650), ("Machining", 820), ("Assembly", 600), ("Painting", 480),
                      ("Testing & FAT", 1100), ("Site commissioning", 1800), ("Project management", 1700)]:
        labour_rates.append({"resource_type": res, "effective_from": f"{yr}-04-01",
                             "cost_rate_inr_per_hr": round(base * infl), "sell_rate_inr_per_hr": round(base * infl * 1.55)})
labour_rates = pd.DataFrame(labour_rates)

# ----------------------------------------------------------------- commodity prices (monthly)
months = pd.date_range("2020-07-01", "2026-09-01", freq="MS")
comm_rows, comm_series = [], {}
for c, (unit, base) in COMMODITIES.items():
    walk = np.cumsum(RNG.normal(0.002, 0.035, len(months)))
    shock = np.where((months >= "2024-03-01") & (months <= "2024-10-01"), 0.08, 0) if c in ("Copper", "Steel HR coil") else 0
    series = base * np.exp(walk + shock)
    comm_series[c] = pd.Series(series, index=months)
    for m, v in zip(months, series):
        comm_rows.append({"month": m.date(), "commodity": c, "price": round(v, 2), "unit": unit,
                          "source": "Synthetic (replace with MCX/LME/supplier index)"})
commodity_prices = pd.DataFrame(comm_rows)


def comm_index(c, d):
    s = comm_series[c]
    d = pd.Timestamp(d).to_period("M").to_timestamp()
    d = min(max(d, s.index[0]), s.index[-1])
    return s.loc[d] / s.iloc[0]


# ----------------------------------------------------------------- RFQs
fam = pick(PRODUCT_FAMILIES.family_id.values, N_RFQ, p=[.24, .18, .16, .16, .16, .10])
fam_df = PRODUCT_FAMILIES.set_index("family_id")
cust = pick(customers.customer_id.values, N_RFQ)
recv = pd.Series(rand_date(START, END - pd.Timedelta(days=20), N_RFQ)).sort_values().values
cap_frac = RNG.beta(2, 3, N_RFQ)
capacity = [round(fam_df.loc[f].capacity_min + (fam_df.loc[f].capacity_max - fam_df.loc[f].capacity_min) * c, 1)
            for f, c in zip(fam, cap_frac)]
complexity = np.clip(np.round(RNG.normal(3, 1.1, N_RFQ)), 1, 5).astype(int)
custom_pct = np.clip(RNG.normal(35 + 9 * complexity, 12), 5, 95).round()
power_kw = np.round(np.where(np.isin(fam, ["PF03"]), RNG.uniform(0, 30, N_RFQ),
                             5 + 600 * cap_frac * RNG.uniform(.5, 1.4, N_RFQ)), 1)
weight_t = np.round((2 + 80 * cap_frac) * (0.6 + 0.2 * complexity) * RNG.uniform(.7, 1.3, N_RFQ), 1)
qty = pick([1, 1, 1, 1, 2, 2, 3, 4], N_RFQ)
pages = np.clip((20 + 25 * complexity + RNG.normal(0, 25, N_RFQ)).astype(int), 4, 420)

rfqs = pd.DataFrame({
    "rfq_id": [f"RFQ-{y}-{n:04d}" for y, n in zip(pd.to_datetime(recv).year,
                                                  pd.Series(pd.to_datetime(recv).year).groupby(pd.to_datetime(recv).year).cumcount() + 1)],
    "customer_id": cust,
    "received_date": pd.to_datetime(recv).date,
    "bid_due_date": (pd.to_datetime(recv) + pd.to_timedelta(RNG.integers(7, 45, N_RFQ), unit="D")).date,
    "family_id": fam,
    "rfq_title": [f"{fam_df.loc[f].family_name} - {c} {fam_df.loc[f].capacity_unit}" for f, c in zip(fam, capacity)],
    "channel": pick(["Email", "Customer portal", "Tender portal (GeM/eProc)", "Sales rep", "EPC contractor"], N_RFQ, p=[.35, .15, .15, .2, .15]),
    "tender_type": pick(["Budgetary", "Firm", "Re-quote"], N_RFQ, p=[.3, .6, .1]),
    "end_user_industry": customers.set_index("customer_id").industry.reindex(cust).values,
    "site_region": customers.set_index("customer_id").region.reindex(cust).values,
    "quantity": qty,
    "capacity_value": capacity,
    "capacity_unit": [fam_df.loc[f].capacity_unit for f in fam],
    "installed_power_kw": power_kw,
    "estimated_weight_t": weight_t,
    "primary_material": pick(MATERIALS, N_RFQ),
    "design_code": pick(DESIGN_CODES, N_RFQ),
    "operating_environment": pick(ENVIRONMENTS, N_RFQ),
    "complexity_score_1to5": complexity,
    "customisation_pct": custom_pct,
    "scope_includes_installation": pick(["Y", "N"], N_RFQ, p=[.55, .45]),
    "scope_includes_civil": pick(["Y", "N"], N_RFQ, p=[.15, .85]),
    "required_delivery_weeks": np.clip((10 + 3 * complexity + 0.1 * weight_t + RNG.normal(0, 5, N_RFQ)).astype(int), 6, 60),
    "ld_clause_pct_per_week": pick([0, 0.5, 0.5, 1.0, 1.0, 1.5, 2.0], N_RFQ),
    "ld_cap_pct": pick([0, 5, 5, 10, 10, 10, 15, 20], N_RFQ),
    "warranty_months": pick([12, 12, 18, 18, 24, 36], N_RFQ),
    "performance_guarantee": pick(["Y", "N"], N_RFQ, p=[.4, .6]),
    "payment_terms": pick(["30/60/10 advance/dispatch/commissioning", "10/80/10", "100% against delivery",
                           "20% advance, 80% LC", "Milestone-based", "Net 90 after delivery"], N_RFQ),
    "inco_terms": pick(["EXW", "FOR site", "CIF", "DAP", "FOB"], N_RFQ, p=[.15, .5, .1, .15, .1]),
    "currency": pick(["INR", "INR", "INR", "INR", "USD", "EUR", "AED"], N_RFQ),
    "rfq_document_pages": pages,
    "num_drawings_attached": np.clip((complexity * 3 + RNG.normal(0, 3, N_RFQ)).astype(int), 0, 40),
    "competitors_invited": RNG.integers(1, 7, N_RFQ),
    "spec_clarity_1to5": np.clip(np.round(RNG.normal(3.2, 1, N_RFQ)), 1, 5).astype(int),
    "rfq_status": "Closed",
})
for rid, fix in PINNED_RFQS.items():
    m = rfqs.rfq_id == rid
    assert m.sum() == 1, rid
    for k, v in fix.items():
        rfqs.loc[m, k] = pd.Timestamp(v).date() if k.endswith("_date") else v
    cinfo = customers.set_index("customer_id").loc[fix["customer_id"]]
    rfqs.loc[m, ["end_user_industry", "site_region", "capacity_unit"]] = [cinfo.industry, cinfo.region, fam_df.loc[fix["family_id"]].capacity_unit]
rfqs.loc[pd.to_datetime(rfqs.received_date) > pd.Timestamp("2026-08-15"), "rfq_status"] = "Open"

# ----------------------------------------------------------------- RFQ documents (metadata only)
doc_rows, did = [], 1
for r in rfqs.itertuples():
    docs = [("Technical specification", "pdf"), ("Commercial terms & conditions", "pdf"), ("Scope of supply", "docx")]
    docs += [("General arrangement drawing", "dwg")] * min(r.num_drawings_attached, 3)
    docs += [("Data sheet", "xlsx")] if RNG.random() < .6 else []
    docs += [("Vendor list / approved makes", "pdf")] if RNG.random() < .5 else []
    docs += [("Bill of quantities", "xlsx")] if RNG.random() < .4 else []
    docs += [("Site layout / P&ID", "pdf")] if RNG.random() < .45 else []
    for k, (dt, fmt) in enumerate(docs, 1):
        doc_rows.append({"document_id": f"D{did:06d}", "rfq_id": r.rfq_id, "document_type": dt, "file_format": fmt,
                         "file_name": f"{r.rfq_id}_{dt.split()[0].lower()}_{k}.{fmt}",
                         "pages": int(max(1, r.rfq_document_pages * RNG.uniform(.05, .45))) if fmt in ("pdf", "docx") else None,
                         "revision": pick(["R0", "R0", "R1", "R2"]), "received_date": r.received_date,
                         "storage_path": f"/rfq_archive/{pd.Timestamp(r.received_date).year}/{r.rfq_id}/"})
        did += 1
rfq_documents = pd.DataFrame(doc_rows)

# ----------------------------------------------------------------- RFQ requirements (LLM extraction target)
REQ_TEMPLATES = {
    "Performance": ["Rated capacity shall be {cap} {unit} continuous", "Availability shall be not less than 98%",
                    "Noise level shall not exceed 85 dB(A) at 1 m", "Turndown ratio 3:1 without loss of control"],
    "Design code": ["Design, fabrication and testing as per {code}", "Seismic design per IS 1893 Zone {z}",
                    "Wind load per IS 875 Part 3"],
    "Material": ["Wetted parts in {mat}", "All structural steel to IS 2062 E250 minimum", "No asbestos or lead-based paint"],
    "Electrical": ["Motors IE3 efficiency class minimum", "Motors and instruments suitable for {env}",
                   "Control voltage 24 V DC", "Power supply 415 V, 3 ph, 50 Hz"],
    "Automation": ["PLC from approved makes only (Siemens/Rockwell/Schneider)", "Integration with plant DCS over Profinet / Modbus TCP",
                   "Remote monitoring capability required"],
    "Approved makes": ["Bearings: SKF/FAG/Timken only", "Gearboxes: approved makes list Annexure-3", "Valves: approved vendor list only"],
    "Testing & inspection": ["Factory acceptance test witnessed by customer", "Third-party inspection by TPI agency",
                             "Hydrotest at 1.5x design pressure", "Radiography 100% on butt welds"],
    "Documentation": ["As-built drawings and O&M manuals in 3 hard copies + soft copy", "Material test certificates EN 10204 3.1",
                      "Spare parts list with prices for 2 years operation"],
    "Commercial": ["Liquidated damages {ld}% per week capped at {cap_ld}%", "Warranty {w} months from commissioning",
                   "Performance bank guarantee 10% of contract value", "Price firm until completion, no escalation"],
    "Site & installation": ["Supervision of erection and commissioning included", "Training of customer operators 5 days",
                            "Site works to comply with customer EHS policy"],
}
PREFIX = ["", "", "", "The bidder shall ensure that ", "Vendor to note: ", "It is mandatory that ", "Note: ", "Contractor shall comply: "]
SUFFIX = ["", "", "", " (refer Annexure-2).", " unless otherwise approved by the Purchaser.", " - no deviation permitted.", " as per site conditions."]


def phrase(t):
    pre = pick(PREFIX)
    t = (pre + (t[0].lower() + t[1:] if pre else t)).rstrip(".") + pick(SUFFIX)
    return t if t.endswith(".") else t + "."


def respond(cat, txt, r):
    """Our compliance response, driven by requirement type and RFQ context (gives a learnable signal)."""
    p_dev, p_clar, p_na = 0.04, 0.10, 0.04
    if cat == "Commercial":
        if "Liquidated" in txt:
            p_dev = 0.75 if (r.ld_cap_pct >= 15 or r.ld_clause_pct_per_week >= 1.5) else 0.15
        elif "Warranty" in txt:
            p_dev = 0.6 if r.warranty_months >= 24 else 0.05
        elif "firm" in txt:
            p_dev = 0.45 if r.required_delivery_weeks > 24 else 0.15
        elif "bank guarantee" in txt:
            p_dev = 0.25
    elif cat in ("Electrical", "Automation"):
        if "suitable for" in txt and ("hazard" in txt.lower() or "corrosive" in txt.lower()):
            p_clar = 0.45
        if "DCS" in txt or "Remote" in txt:
            p_clar = 0.4
    elif cat == "Testing & inspection" and ("Third-party" in txt or "Radiography" in txt):
        p_clar, p_dev = 0.3, 0.15
    elif cat == "Approved makes":
        p_dev = 0.2 if r.customisation_pct > 70 else 0.05
    elif cat == "Site & installation" and r.scope_includes_installation == "N":
        p_na = 0.6
    if r.spec_clarity_1to5 <= 2:
        p_clar += 0.2
    u = RNG.random()
    if u < p_na:
        return "Not applicable", "", "None"
    if u < p_na + p_dev:
        dev = {"Commercial": pick(["Propose LD cap of 10%", "Offer 18 months from commissioning", "Request price variation clause",
                                   "Propose PBG of 5%"]),
               "Approved makes": "Offer alternate make of equivalent quality",
               "Testing & inspection": "Witness FAT at our works only"}.get(cat, pick(["Propose equivalent material", "Request extended delivery"]))
        return "Deviation", dev, pick(["Medium", "High"], p=[.55, .45])
    if u < p_na + p_dev + p_clar:
        return "Comply with clarification", pick(["Need site data", "Confirm interface scope", "Clarify DCS protocol and signal list",
                                                  "Confirm hazardous area classification", "Confirm scope split with EPC"]), pick(["None", "Low", "Medium"], p=[.3, .5, .2])
    return "Comply", "", pick(["None", "Low"], p=[.8, .2])


req_rows, qid = [], 1
for r in rfqs.itertuples():
    n = int(np.clip(4 + 2 * r.complexity_score_1to5 + RNG.normal(0, 2), 4, 22))
    cats = pick(list(REQ_TEMPLATES), n)
    for c in cats:
        txt = pick(REQ_TEMPLATES[c]).format(cap=r.capacity_value, unit=r.capacity_unit, code=r.design_code, z=pick(["III", "IV"]),
                                            mat=r.primary_material, env=r.operating_environment.lower(), ld=r.ld_clause_pct_per_week,
                                            cap_ld=r.ld_cap_pct, w=r.warranty_months)
        status, dev, cost_imp = respond(c, txt, r)
        txt = phrase(txt)
        req_rows.append({"requirement_id": f"REQ{qid:07d}", "rfq_id": r.rfq_id, "category": c, "requirement_text": txt,
                         "source_document": pick(["Technical specification", "Commercial terms & conditions", "Scope of supply"]),
                         "source_page": int(RNG.integers(1, max(2, r.rfq_document_pages))),
                         "mandatory": pick(["Y", "Y", "N"]), "compliance_status": status, "deviation_or_clarification": dev,
                         "cost_impact": cost_imp})
        qid += 1
rfq_requirements = pd.DataFrame(req_rows)

# ----------------------------------------------------------------- shop capacity (weekly)
weeks = pd.date_range("2021-01-04", "2026-12-28", freq="W-MON")
wc = [("WC-FAB", "Fabrication bay", 2400), ("WC-MCH", "Machining", 900), ("WC-ASM", "Assembly", 1600),
      ("WC-ELE", "Electrical & panel shop", 800), ("WC-TST", "Testing & FAT", 300), ("WC-ENG", "Design office", 1800)]
cap_rows = []
ar = {code: 0.0 for code, _, _ in wc}
trend = {code: 0.0 for code, _, _ in wc}
for w in weeks:
    season = 0.08 * np.sin(2 * np.pi * (w.dayofyear / 365) - 1)
    for code, name, cap in wc:
        ar[code] = 0.93 * ar[code] + RNG.normal(0, 0.045)          # persistent order-book swings
        load = cap * np.clip(0.82 + season + ar[code] + RNG.normal(0, .04), 0.35, 1.35)
        cap_rows.append({"week_start": w.date(), "work_center_id": code, "work_center_name": name,
                         "available_hrs": cap, "loaded_hrs": round(load if w <= pd.Timestamp("2026-10-12") else load * RNG.uniform(.4, .9)),
                         "utilisation_pct": None, "is_forecast": "Y" if w > pd.Timestamp("2026-10-12") else "N"})
shop_capacity = pd.DataFrame(cap_rows)
shop_capacity["utilisation_pct"] = (100 * shop_capacity.loaded_hrs / shop_capacity.available_hrs).round(1)

FAB_UTIL = shop_capacity[shop_capacity.work_center_id == "WC-FAB"].set_index("week_start").utilisation_pct / 100


def fab_util(d):
    """Average fabrication-bay utilisation over the 12 weeks after date d (proxy for shop congestion)."""
    w = (pd.Timestamp(d) - pd.Timedelta(days=pd.Timestamp(d).weekday())).date()
    i = FAB_UTIL.index.searchsorted(w)
    return float(FAB_UTIL.iloc[i:i + 12].mean()) if i < len(FAB_UTIL) else 0.8


# ----------------------------------------------------------------- bid/no-bid decision + quotes
cust_idx = customers.set_index("customer_id")
est_ids = estimators.estimator_id.values
est_bias = estimators.set_index("estimator_id")._bias

# "true" cost drivers per family (multipliers)
FAM_COST = {"PF01": 1.0, "PF02": 1.25, "PF03": 0.9, "PF04": 1.5, "PF05": 1.1, "PF06": 1.2}

quotes, qlines, outcomes, projects, pbom, sq_rows = [], [], [], [], [], []
line_items = items.copy()
item_by_cat = {c: g for c, g in line_items.groupby("item_category")}

FAM_BOM_CATS = {
    "PF01": ["Structural", "Drive", "Drive", "Mechanical", "Mechanical", "Electrical", "Automation", "Finish", "Consumable"],
    "PF02": ["Structural", "Fluid", "Fluid", "Fluid", "Instrumentation", "Instrumentation", "Electrical", "Automation", "Finish"],
    "PF03": ["Plate", "Plate", "Structural", "Fluid", "Instrumentation", "Finish", "Subcontract", "Consumable"],
    "PF04": ["Structural", "Drive", "Mechanical", "Mechanical", "Automation", "Automation", "Instrumentation", "Electrical", "Consumable"],
    "PF05": ["Electrical", "Electrical", "Electrical", "Automation", "Automation", "Structural", "Consumable"],
    "PF06": ["Structural", "Structural", "Drive", "Drive", "Mechanical", "Electrical", "Automation", "Finish", "Subcontract"],
}


RATE = {(r.resource_type, r.effective_from[:4]): r.cost_rate_inr_per_hr for r in labour_rates.itertuples()}


def rate(resource, d):
    d = pd.Timestamp(d)
    yr = min(max(d.year - (1 if d.month < 4 else 0), 2020), 2026)
    return RATE[(resource, str(yr))]


REQ_COUNT = rfq_requirements.rfq_id.value_counts()
ITEMS_IDX = items.set_index("item_id")
SUP_IDX = suppliers.set_index("supplier_id")
SUP_BY_CAT = {c: g.supplier_id.values for c, g in suppliers.groupby("category")}


qn = 1
pn = 1
for r in rfqs.itertuples():
    c = cust_idx.loc[r.customer_id]
    # bid / no-bid
    util_q = fab_util(r.received_date)
    z = (-3.2 + 1.0 * (r.spec_clarity_1to5 <= 2) + 1.4 * (c.credit_rating == "D") + 0.5 * (c.credit_rating == "C")
         + 0.8 * (r.ld_cap_pct >= 15) + 7.0 * max(0.0, util_q - 0.85) + 0.6 * (r.complexity_score_1to5 == 5)
         + 0.25 * (r.competitors_invited - 3) + 0.7 * (r.required_delivery_weeks < 8 + 2 * r.complexity_score_1to5)
         - 0.8 * (c.segment == "Strategic") + 0.5 * (r.tender_type == "Budgetary"))
    nobid_p = 1 / (1 + np.exp(-z))
    if RNG.random() < nobid_p and r.rfq_id not in FORCED_OUTCOME:
        outcomes.append({"rfq_id": r.rfq_id, "quote_id": None, "bid_decision": "No-bid",
                         "no_bid_reason": pick(["Out of capability", "Unacceptable commercial terms", "Delivery not achievable",
                                                "Low win probability", "Credit risk", "Engineering capacity full"]),
                         "outcome": "No-bid", "decision_date": r.bid_due_date, "loss_reason": None, "winning_competitor_id": None,
                         "winning_price_inr_if_known": None, "final_order_value_inr": None})
        continue

    est = pick(est_ids)
    qdate = pd.Timestamp(r.received_date) + pd.Timedelta(days=int(RNG.integers(5, max(6, (pd.Timestamp(r.bid_due_date) - pd.Timestamp(r.received_date)).days + 3))))
    m = FAM_COST[r.family_id]
    size = (r.estimated_weight_t ** 0.85) * (1 + 0.004 * r.installed_power_kw)
    cx = 1 + 0.18 * (r.complexity_score_1to5 - 1) + 0.004 * r.customisation_pct
    env = 1.12 if ("Hazard" in r.operating_environment or "Corros" in r.operating_environment or "saline" in r.operating_environment) else 1.0
    mat = {"SS304": 1.35, "SS316L": 1.5, "Duplex 2205": 1.8, "SA516 Gr70": 1.1, "Aluminium 6061": 1.25}.get(r.primary_material, 1.0)

    # TRUE (actual) quantities
    true_material = 95000 * m * size * mat * env * r.quantity * comm_index("Steel HR coil", qdate) ** 0.5 * RNG.lognormal(0, .12)
    true_eng = (60 + 55 * size ** 0.6 * cx) * (1 + 0.15 * (r.quantity - 1) ** 0.3) * RNG.lognormal(0, .15)
    true_fab = (25 * r.estimated_weight_t * m * cx * r.quantity) * RNG.lognormal(0, .15)
    true_asm = (0.45 * true_fab + 30 * r.complexity_score_1to5 * r.quantity) * RNG.lognormal(0, .15)
    true_test = (16 + 12 * r.complexity_score_1to5 + (40 if "Third-party" in r.design_code else 0)) * RNG.lognormal(0, .2)
    true_sub = true_material * (0.08 + 0.04 * (r.scope_includes_installation == "Y") + 0.15 * (r.scope_includes_civil == "Y")) * RNG.lognormal(0, .2)

    # ESTIMATES: bias from estimator experience + systematic underestimation of complex/custom jobs
    bias = est_bias[est] * (1 - 0.025 * (r.complexity_score_1to5 - 3) - 0.0012 * (r.customisation_pct - 40))
    e_mat = true_material * bias * RNG.lognormal(0, .06)
    e_eng = true_eng * bias * RNG.lognormal(0, .1)
    e_fab = true_fab * bias * RNG.lognormal(0, .08)
    e_asm = true_asm * bias * RNG.lognormal(0, .08)
    e_test = true_test * bias * RNG.lognormal(0, .1)
    e_sub = true_sub * bias * RNG.lognormal(0, .1)

    eng_rate, fab_rate = rate("Design engineering", qdate), rate("Fabrication", qdate)
    asm_rate, test_rate = rate("Assembly", qdate), rate("Testing & FAT", qdate)
    e_lab = e_eng * eng_rate + e_fab * fab_rate + e_asm * asm_rate + e_test * test_rate
    contingency = float(pick([3, 5, 5, 7.5, 10])) + (2.5 if r.complexity_score_1to5 >= 4 else 0)
    total_cost = (e_mat + e_lab + e_sub) * (1 + contingency / 100)
    margin = float(np.clip(RNG.normal({"Strategic": 18, "Key": 20, "Standard": 23, "New": 21}[c.segment], 4), 8, 35))
    price = total_cost / (1 - margin / 100)
    lead = int(max(6, r.required_delivery_weeks + 8 * max(0.0, util_q - 0.8) + RNG.normal(1.0, 2.5)))
    effort = round(float(8 + 14 * r.complexity_score_1to5 + 0.12 * r.rfq_document_pages + RNG.normal(0, 8)), 1)
    revisions = int(pick([0, 0, 0, 1, 1, 2, 3]))

    quote_id = f"Q{qn:05d}"
    qn += 1
    quotes.append({
        "quote_id": quote_id, "rfq_id": r.rfq_id, "revision": f"R{revisions}", "quote_date": qdate.date(),
        "estimator_id": est, "pricing_model": pick(["Fixed price", "Fixed price", "Fixed price", "T&M for site works", "Price + escalation clause"]),
        "est_material_cost_inr": round(e_mat), "est_engineering_hrs": round(e_eng, 1), "est_fabrication_hrs": round(e_fab, 1),
        "est_assembly_hrs": round(e_asm, 1), "est_testing_hrs": round(e_test, 1), "est_labour_cost_inr": round(e_lab),
        "est_subcontract_cost_inr": round(e_sub), "contingency_pct": contingency, "est_total_cost_inr": round(total_cost),
        "target_margin_pct": round(margin, 1), "quoted_price_inr": round(price, -3), "discount_pct": 0.0,
        "quoted_lead_time_weeks": lead, "validity_days": int(pick([30, 45, 60, 90])),
        "quote_preparation_hrs": max(4, effort), "num_deviations_taken": int(REQ_COUNT.get(r.rfq_id, 0) * RNG.uniform(0, .25)),
        "similar_past_quote_ref": None,
        "approval_level": "Director" if price > 3e7 else ("Head of Sales" if price > 8e6 else "Sales Manager"),
    })

    # quote lines (estimated BOM)
    cats = FAM_BOM_CATS[r.family_id]
    lines = []
    budget = e_mat
    weights = RNG.dirichlet(np.ones(len(cats)) * 2)
    for ln, (cat, w) in enumerate(zip(cats, weights), 1):
        it = item_by_cat[cat].sample(1, random_state=int(RNG.integers(1e9))).iloc[0]
        ext = budget * w
        q = max(1, round(ext / it.standard_cost_inr, 1 if it.uom in ("kg", "m", "m2") else 0))
        unit = ext / q
        lines.append({"quote_id": quote_id, "line_no": ln, "item_id": it.item_id, "line_description": it.item_description,
                      "line_type": "Material", "qty": q, "uom": it.uom, "est_unit_cost_inr": round(unit, 2), "est_ext_cost_inr": round(ext)})
    for ln_off, (res, hrs, rr) in enumerate([("Design engineering", e_eng, eng_rate), ("Fabrication", e_fab, fab_rate),
                                             ("Assembly", e_asm, asm_rate), ("Testing & FAT", e_test, test_rate)], len(lines) + 1):
        lines.append({"quote_id": quote_id, "line_no": ln_off, "item_id": None, "line_description": res, "line_type": "Labour",
                      "qty": round(hrs, 1), "uom": "hr", "est_unit_cost_inr": rr, "est_ext_cost_inr": round(hrs * rr)})
    lines.append({"quote_id": quote_id, "line_no": len(lines) + 1, "item_id": None, "line_description": "Subcontract / site services",
                  "line_type": "Subcontract", "qty": 1, "uom": "lot", "est_unit_cost_inr": round(e_sub), "est_ext_cost_inr": round(e_sub)})
    qlines.extend(lines)

    # supplier quotes for the 3 biggest bought-out lines
    mat_lines = sorted([l for l in lines if l["line_type"] == "Material"], key=lambda x: -x["est_ext_cost_inr"])[:3]
    for l in mat_lines:
        it = ITEMS_IDX.loc[l["item_id"]]
        if it.make_buy != "Buy":
            continue
        sups = SUP_BY_CAT[SUP_IDX.loc[it.preferred_supplier_id].category] if it.preferred_supplier_id else suppliers.supplier_id.values
        for s in pick(sups, min(len(sups), int(RNG.integers(1, 4)))):
            sq_rows.append({"supplier_quote_id": f"SQ{len(sq_rows) + 1:06d}", "rfq_id": r.rfq_id, "quote_id": quote_id, "supplier_id": s,
                            "item_id": l["item_id"], "qty": l["qty"], "quoted_unit_price_inr": round(l["est_unit_cost_inr"] * (1 + 0.05 * (SUP_IDX.loc[s].quality_rating - 3.75))
                                                            * {"Germany": 1.18, "Japan": 1.12, "Italy": 1.1, "USA": 1.15, "China": 0.88}.get(SUP_IDX.loc[s].country, 1.0)
                                                            * comm_index(it.commodity_driver, qdate) ** 0.3 * RNG.lognormal(0, .06), 2),
                            "quoted_lead_time_days": int(max(3, (0.5 * it.standard_lead_time_days + 0.5 * SUP_IDX.loc[s].avg_lead_time_days)
                                                                 * (1 + 0.006 * (95 - SUP_IDX.loc[s].on_time_delivery_pct)) * (1.0 + 0.6 * max(0.0, util_q - 0.85))
                                                                 * RNG.lognormal(0, .12))),
                            "quote_received_date": (qdate - pd.Timedelta(days=int(RNG.integers(1, 10)))).date(),
                            "valid_until": (qdate + pd.Timedelta(days=int(pick([15, 30, 45])))).date(),
                            "source_format": pick(["PDF email attachment", "Excel", "Supplier portal", "Phone (manual entry)"])})

    # outcome
    if r.rfq_status == "Open":
        outcomes.append({"rfq_id": r.rfq_id, "quote_id": quote_id, "bid_decision": "Bid", "no_bid_reason": None, "outcome": "Pending",
                         "decision_date": None, "loss_reason": None, "winning_competitor_id": None, "winning_price_inr_if_known": None,
                         "final_order_value_inr": None})
        continue
    comp = competitors.sample(1, random_state=int(RNG.integers(1e9))).iloc[0]
    comp_strong = 0.95 if r.family_id in comp.strong_families else 1.0
    comp_price = comp_strong * (true_material + true_eng * eng_rate + true_fab * fab_rate + true_asm * asm_rate + true_test * test_rate + true_sub) \
        * comp.typical_price_index * 1.22 * RNG.lognormal(0, .08)
    ratio = price / comp_price
    logit = (1.0 - 9.0 * (ratio - 1.0) - 0.07 * (lead - r.required_delivery_weeks) - 0.18 * (r.competitors_invited - 3)
             + {"Strategic": 1.0, "Key": 0.5, "Standard": 0, "New": -0.5}[c.segment] + (-0.6 if r.tender_type == "Budgetary" else 0)
             - 0.05 * revisions - 1.9)
    pwin = 1 / (1 + np.exp(-logit))
    roll = RNG.random()
    if r.rfq_id in FORCED_OUTCOME:
        outcome = FORCED_OUTCOME[r.rfq_id]
        loss = "Commercial terms" if outcome == "Lost" else None
    elif r.tender_type == "Budgetary" and RNG.random() < .45:
        outcome, loss = "Cancelled / no decision", pick(["Project deferred", "Budget not approved", "Customer went silent"])
    elif roll < pwin:
        outcome, loss = "Won", None
    else:
        outcome = "Lost"
        if ratio > 1.04:
            loss = "Price too high"
        elif lead > r.required_delivery_weeks + 3:
            loss = "Delivery too long"
        else:
            loss = pick(["Technical non-compliance", "Relationship / incumbent", "Commercial terms", "Price too high", "Lower-spec competitor"])
    disc = float(np.clip(RNG.normal(0.5 + 0.18 * (margin - 15) + 0.7 * (r.competitors_invited - 3)
                                    + {"Strategic": 2.0, "Key": 1.0, "Standard": 0.0, "New": 0.5}[c.segment]
                                    + 1.0 * (c.procurement_style == "EPC contractor"), 1.0), 0, 12)) if outcome == "Won" else 0.0
    final_val = round(price * (1 - disc / 100), -3) if outcome == "Won" else None
    quotes[-1]["discount_pct"] = round(disc, 1)
    outcomes.append({"rfq_id": r.rfq_id, "quote_id": quote_id, "bid_decision": "Bid", "no_bid_reason": None, "outcome": outcome,
                     "decision_date": (qdate + pd.Timedelta(days=int(RNG.integers(14, 120)))).date(),
                     "loss_reason": loss, "winning_competitor_id": comp.competitor_id if outcome == "Lost" else None,
                     "winning_price_inr_if_known": round(comp_price, -3) if (outcome == "Lost" and RNG.random() < .4) else None,
                     "final_order_value_inr": final_val})

    # project actuals for won quotes
    if outcome == "Won":
        order_date = pd.Timestamp(outcomes[-1]["decision_date"])
        drift = comm_index("Steel HR coil", order_date + pd.Timedelta(weeks=6)) / comm_index("Steel HR coil", qdate)
        a_mat = true_material * drift ** 0.6
        change_orders = final_val * float(pick([0, 0, 0, 0.02, 0.05, 0.08, 0.12])) * RNG.uniform(.5, 1.2)
        planned = order_date + pd.Timedelta(weeks=lead)
        util_o = fab_util(order_date)
        slip_w = max(0.0, RNG.normal(-2.5 + 0.9 * (r.complexity_score_1to5 - 1) + 0.035 * r.customisation_pct
                                     + 25 * (util_o - 0.82) + 2.0 * (r.spec_clarity_1to5 <= 2)
                                     + 0.6 * max(0, r.required_delivery_weeks - lead + 4), 1.8))
        actual_del = planned + pd.Timedelta(weeks=round(slip_w, 1))
        status = "Closed" if actual_del < pd.Timestamp("2026-09-30") else "In execution"
        ld_paid = 0.0
        if r.ld_clause_pct_per_week and slip_w > 1:
            ld_paid = final_val * min(r.ld_cap_pct, r.ld_clause_pct_per_week * (slip_w - 1)) / 100 * float(pick([1, 1, 0.5, 0]))
        a_lab = true_eng * eng_rate + true_fab * fab_rate + true_asm * asm_rate + true_test * test_rate
        warranty = final_val * float(pick([0, 0, 0, 0, 0.005, 0.01, 0.03])) if status == "Closed" else 0.0
        a_total = a_mat + a_lab + true_sub + ld_paid + warranty
        project_id = f"P{pn:04d}"
        pn += 1
        projects.append({
            "project_id": project_id, "quote_id": quote_id, "rfq_id": r.rfq_id, "customer_id": r.customer_id, "order_date": order_date.date(),
            "order_value_inr": final_val, "change_order_value_inr": round(change_orders), "planned_delivery_date": planned.date(),
            "actual_delivery_date": actual_del.date() if status == "Closed" else None, "project_status": status,
            "actual_material_cost_inr": round(a_mat) if status == "Closed" else None,
            "actual_engineering_hrs": round(true_eng, 1) if status == "Closed" else None,
            "actual_fabrication_hrs": round(true_fab, 1) if status == "Closed" else None,
            "actual_assembly_hrs": round(true_asm, 1) if status == "Closed" else None,
            "actual_testing_hrs": round(true_test, 1) if status == "Closed" else None,
            "actual_labour_cost_inr": round(a_lab) if status == "Closed" else None,
            "actual_subcontract_cost_inr": round(true_sub) if status == "Closed" else None,
            "ld_paid_inr": round(ld_paid) if status == "Closed" else None,
            "warranty_cost_inr": round(warranty) if status == "Closed" else None,
            "actual_total_cost_inr": round(a_total) if status == "Closed" else None,
            "actual_margin_pct": round(100 * (final_val + change_orders - a_total) / (final_val + change_orders), 1) if status == "Closed" else None,
            "num_engineering_changes": int(RNG.poisson(0.3 + 0.6 * r.complexity_score_1to5 + 0.04 * r.customisation_pct + 2.0 * (r.spec_clarity_1to5 <= 2))),
            "root_cause_of_overrun": (pick(["Scope change not priced", "Design iterations", "Material price increase", "Vendor delay",
                                            "Rework / quality", "Underestimated fabrication", "Site delays (customer)"])
                                      if status == "Closed" and a_total > total_cost * 1.05 else None),
        })
        # actual BOM: estimated lines with qty / price drift + occasional added items
        if status == "Closed":
            for l in [x for x in lines if x["line_type"] == "Material"]:
                pbom.append({"project_id": project_id, "item_id": l["item_id"], "line_description": l["line_description"],
                             "actual_qty": round(l["qty"] * RNG.lognormal(0.02, .08), 1), "uom": l["uom"],
                             "actual_unit_cost_inr": round(l["est_unit_cost_inr"] / bias * drift ** 0.6 * RNG.lognormal(0, .05), 2),
                             "added_after_quote": "N"})
            if RNG.random() < min(0.9, 0.02 + 0.09 * r.complexity_score_1to5 + 0.004 * r.customisation_pct + 0.2 * (r.spec_clarity_1to5 <= 2)):
                it = items.sample(1, random_state=int(RNG.integers(1e9))).iloc[0]
                pbom.append({"project_id": project_id, "item_id": it.item_id, "line_description": it.item_description,
                             "actual_qty": 1 if it.uom == "nos" else round(RNG.uniform(10, 200), 1), "uom": it.uom,
                             "actual_unit_cost_inr": round(it.standard_cost_inr * RNG.lognormal(0, .1), 2), "added_after_quote": "Y"})

quotes = pd.DataFrame(quotes)
quote_lines = pd.DataFrame(qlines)
quote_outcomes = pd.DataFrame(outcomes)
projects = pd.DataFrame(projects)
project_bom = pd.DataFrame(pbom)
supplier_quotes = pd.DataFrame(sq_rows)

# similar past quote reference: earlier quote, same family, nearest capacity
qq = quotes[["quote_id", "rfq_id", "quote_date"]].merge(rfqs[["rfq_id", "family_id", "capacity_value"]], on="rfq_id")
qq["qd"] = pd.to_datetime(qq.quote_date)
ref_map = {}
for fam_id, g in qq.groupby("family_id"):
    g = g.sort_values("qd")
    ids, caps, dates = g.quote_id.values, g.capacity_value.values.astype(float), g.qd.values
    for i in range(len(g)):
        prior = np.nonzero(dates[:i] < dates[i])[0]
        if len(prior) and RNG.random() < .7:
            prior = prior[-300:]
            ref_map[ids[i]] = ids[prior[np.argmin(np.abs(caps[prior] - caps[i]))]]
refs = [ref_map.get(q) for q in quotes.quote_id]
quotes["similar_past_quote_ref"] = refs

# ----------------------------------------------------------------- contract clauses (risk review target)
clause_rows = []
for r in rfqs.itertuples():
    clauses = [
        ("Liquidated damages", f"{r.ld_clause_pct_per_week}% per week, cap {r.ld_cap_pct}%", "0.5% per week, cap 5%",
         "High" if r.ld_cap_pct >= 15 or r.ld_clause_pct_per_week >= 1.5 else ("Medium" if r.ld_cap_pct >= 10 else "Low")),
        ("Warranty", f"{r.warranty_months} months", "12 months from commissioning or 18 from dispatch",
         "High" if r.warranty_months >= 36 else ("Medium" if r.warranty_months >= 24 else "Low")),
        ("Payment terms", r.payment_terms, "30/60/10", "High" if "Net 90" in r.payment_terms or "100% against" in r.payment_terms else "Low"),
        ("Performance guarantee", "Required" if r.performance_guarantee == "Y" else "Not required", "Capacity only, no penalty",
         "Medium" if r.performance_guarantee == "Y" else "Low"),
        ("Limitation of liability", pick(["Unlimited", "100% of contract value", "Not specified", "50% of contract value"]),
         "Capped at 100% of contract value, excl. consequential", None),
        ("Price escalation", pick(["Firm price", "Firm price", "PVC clause allowed"]), "PVC on steel and copper", None),
    ]
    for ct, cust_pos, our_pos, risk in clauses:
        if risk is None:
            risk = "High" if cust_pos in ("Unlimited", "Firm price") else ("Medium" if cust_pos == "Not specified" else "Low")
        clause_rows.append({"rfq_id": r.rfq_id, "clause_type": ct, "customer_position": cust_pos,
                            "our_standard_position": our_pos, "risk_level": risk,
                            "accepted_as_is": "Y" if risk == "Low" else pick(["Y", "N", "N"])})
contract_clauses = pd.DataFrame(clause_rows)

estimators = estimators.drop(columns="_bias")

# ----------------------------------------------------------------- nothing may be decided after the data cut-off
CUTOFF = "2026-10-16"
late = quote_outcomes.decision_date.notna() & (quote_outcomes.decision_date.astype(str) > CUTOFF)
late_q = set(quote_outcomes.loc[late & (quote_outcomes.outcome == "Won"), "quote_id"])
quote_outcomes.loc[late, ["outcome", "decision_date", "loss_reason", "winning_competitor_id", "winning_price_inr_if_known",
                          "final_order_value_inr"]] = ["Pending", None, None, None, None, None]
quotes.loc[quotes.quote_id.isin(late_q), "discount_pct"] = 0.0
dropped = set(projects.loc[projects.quote_id.isin(late_q), "project_id"])
projects = projects[~projects.project_id.isin(dropped)].reset_index(drop=True)
project_bom = project_bom[~project_bom.project_id.isin(dropped)].reset_index(drop=True) if len(project_bom) else project_bom

# ----------------------------------------------------------------- write
TABLES = {
    "customers": customers, "product_families": PRODUCT_FAMILIES, "item_master": items, "suppliers": suppliers,
    "competitors": competitors, "estimators": estimators, "labour_rates": labour_rates, "commodity_prices": commodity_prices,
    "rfqs": rfqs, "rfq_documents": rfq_documents, "rfq_requirements": rfq_requirements, "contract_clauses": contract_clauses,
    "quotes": quotes, "quote_lines": quote_lines, "supplier_quotes": supplier_quotes, "quote_outcomes": quote_outcomes,
    "projects": projects, "project_bom_actuals": project_bom, "shop_capacity": shop_capacity,
}
if __name__ == "__main__":
    for name, df in TABLES.items():
        df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
        print(f"{name:22s} {len(df):>6d} rows  {df.shape[1]:>3d} cols")
