"""v4: an assumption store the LLM edits, and predictions code checks (D-042).

Modules, in the order information flows:

- `expect` — the small language the model predicts outcomes in, and the exact
  checker that compares a prediction with what the step did.
- `store` — the assumption store: typed, scoped, weighted beliefs with evidence;
  edited by the model, evidenced by code.
- `render` — the frame as a PNG for multimodal models (added to the exact text).
- `prompts` — the two calls: update the store, then act with expectations.
- `agent` — the controller: predict -> act -> check -> update.

The fact layer is v3's (`arc3.v3.facts`).
"""
