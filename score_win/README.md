# score_win: Agent X win-probability Azure Function

Scores a range of prices for one RFQ and recommends the price with the highest
expected margin (`P(win) x (price - total cost)`), only inside the markup range
the model was trained on. Engineering and sales still approve the final price.

## What's in the folder

| File | Purpose |
| --- | --- |
| `function_app.py` | The function: `POST /api/score_win` and `GET /api/health` |
| `train_model.py` | Trains the model and writes `model/win_model.joblib` + `model/model_meta.json` |
| `requirements.txt` | Python packages (pinned so training and Azure use the same versions) |
| `host.json`, `local.settings.json` | Azure Functions runtime settings |
| `.vscode/` | Tells the VS Code Azure Functions extension how to run and deploy |
| `sample_request.json` | Example request body |

## 1. Prerequisites

- VS Code with the **Azure Functions** and **Python** extensions
- **Python 3.11**
- **Azure Functions Core Tools v4** (VS Code offers to install it the first time you press F5)
- An Azure subscription

## 2. Set up and train (in VS Code's terminal)

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_model.py              # synthetic 6,000-RFQ data
# or, with real history:
# python train_model.py --csv bids.csv
```

Training must happen inside this `.venv`, so the scikit-learn version that saves
the model is the one Azure loads it with. The `model/` folder is deployed with the code.

A real-data CSV needs these columns, one row per decided bid:
`markup, lead_time_weeks, ld_percent, warranty_months, clause_count, deviations, repeat_customer, competitors, won`

## 3. Run locally

Press **F5** (or run `func start`). Then:

```bash
curl -X POST http://localhost:7071/api/score_win \
  -H "Content-Type: application/json" -d @sample_request.json
```

## 4. Deploy to Azure

1. Open the Azure panel in VS Code and sign in.
2. **Functions → Create Function App in Azure…** → name it (e.g. `agentx-score-win`),
   runtime **Python 3.11**, pick your region.
   To reuse your existing sandbox app instead, skip this step.
3. **Functions → Deploy to Function App…** → choose the app.
4. Get the key: right-click **score_win** → **Copy Function Url** (it includes `?code=...`).

## 5. Call it

Request (all fields required except the price options):

```json
{
  "total_cost": 1250000,
  "lead_time_weeks": 22,
  "ld_percent": 10,
  "warranty_months": 24,
  "clause_count": 18,
  "deviations": 2,
  "repeat_customer": 1,
  "competitors": 4
}
```

Price options (pick one, or leave out to search the whole trained range):

- `"markup_min": 1.10, "markup_max": 1.30, "step": 0.01`
- `"candidate_prices": [1400000, 1450000, 1500000]`

Response (trimmed):

```json
{
  "recommended": {
    "price": 1475000.0,
    "markup": 1.18,
    "win_probability": 0.6978,
    "expected_margin": 157007.97,
    "in_training_range": true
  },
  "curve": [ { "price": 1312500.0, "markup": 1.05, "win_probability": 0.8589, ... }, ... ],
  "warnings": [],
  "model": { "source": "synthetic", "rows": 6000, "test_auc": 0.805, "win_rate": 0.286 },
  "note": "Recommendation only. Engineering and sales approve the final price."
}
```

- `warnings` lists any input outside the training data (e.g. a 60-week lead time).
- Prices outside the trained markup range are scored but never recommended.

## 6. Call it from the Foundry agent

Add it as an **OpenAPI tool** on the agent pointing at
`https://<your-app>.azurewebsites.net/api/score_win`, with the function key
sent as the `x-functions-key` header. The agent calls it at step 7 after the
estimator has confirmed total cost and lead time.
