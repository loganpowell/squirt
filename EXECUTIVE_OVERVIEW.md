# Sleuth: Testing for AI Systems

## The Problem

**AI systems fail traditional testing**

- LLMs give different answers each run
- "Is 73% accuracy good?" → You need context
- Manual output inspection wastes hours
- Subjective code reviews miss regressions

---

## The Solution

**Treat testing like monitoring: Track trends, detect regressions, generate insights**

```
🔴 CRITICAL: Accuracy dropped 12%
   → extract_json component
   → Review prompt changes in abc123

⚠️  WARNING: Runtime +45%, tokens doubled
   → Consider caching
```

---

## Design Philosophy

### 1. Insights > Metrics

Auto-generate recommended actions with severity levels

- What happened
- Why it matters
- What to do about it

### 2. Trends > Absolutes

Every metric shows: "Are we improving?"

- Sparklines (last 10 runs)
- % change from previous commit
- Statistical significance

### 3. Health + Details

Two-tier aggregation:

- **System metrics**: "Is the pipeline healthy?" (accuracy: 41.5%)
- **Component metrics**: "Which part is broken?" (extract_json.accuracy: 0.0)

### 4. Effortless DX

```python
@track(metrics=[
    m.expected_match.compare_to_expected("expected", "output"),
    m.custom("cost").compute(lambda i, o: o["tokens"] * 0.001),
])
def my_llm_component(text): ...
```

- Autocomplete in IDE
- Zero config
- Instrument in minutes

#### Anatomy of a Metric

```python
<namespace>.<metric_name>.<operation>(
    <extraction_or_transform>,
    [threshold=<value>]  # For assertions
)

# Examples:
m.runtime_ms.from_output("metadata.runtime")        # Extract from output
m.expected_match.compare_to_expected("exp", "out")  # Compare values
m.custom("cost").compute(lambda i, o: ...)          # Custom transform
llm.relevance.evaluate(evaluator_function)          # External judge
data.node_count.assert_passes(transform, threshold=5)  # Fail CI if < 5
```

**Namespaces**: `m` (built-in), `data` (data quality), `llm` (LLM evaluators)  
**Operations**: `.from_output()`, `.compute()`, `.compare_to_expected()`, `.evaluate()`, `.assert_passes()`

### 5. Automated Gates

```bash
sleuth report check-regression --threshold 5.0
# Fails CI if accuracy drops >5%
# Accounts for normal LLM variance
```

### 6. Visual Debugging

Mermaid treemaps show bottlenecks instantly

- Component hierarchy from AST analysis
- Runtime/memory proportional to box size
- Native GitHub rendering

---

## Key Capabilities

| Capability                | Benefit                                     |
| ------------------------- | ------------------------------------------- |
| **Trend tracking**        | "Are we getting better?" vs "Is this good?" |
| **Regression detection**  | Block PRs that degrade quality              |
| **Treemap visualization** | Find bottlenecks in 5 minutes               |
| **PR comments**           | Data-driven code reviews                    |
| **Custom evaluators**     | Integrate Azure AI, LangChain judges        |
| **Binary matching**       | Clear pass/fail for fuzzy outputs           |

---

## ROI

**4-6 hours saved per engineer per sprint**

| Before                          | After          |
| ------------------------------- | -------------- |
| Manual output inspection: 2-4hr | Automated: 0hr |
| "Is accuracy improving?": 30min | Instant trends |
| Debug bottlenecks: 1-2hr        | Treemaps: 5min |
| Subjective reviews              | PR metrics     |

**5-person team = 100-120 hours/quarter**

---

## Competitive Position

| vs                      | Limitation            | Sleuth Advantage                           |
| ----------------------- | --------------------- | ------------------------------------------ |
| **Traditional testing** | Binary pass/fail      | Trend-based with variance tolerance        |
| **APM tools**           | Production monitoring | Dev-time, component-level, test-integrated |
| **ML trackers**         | Training metrics      | Application metrics, CI/CD gates           |
| **Spreadsheets**        | Manual, error-prone   | Automated, version-controlled              |

---

## Questions Answered

**Developers**: "Did my changes help?" → Compare vs previous commit  
**Managers**: "Are we improving?" → Historical trends + sparklines  
**Product**: "What's our quality?" → System health dashboard  
**DevOps**: "Can we deploy?" → Automated regression gates

---

## Bottom Line

**Transform AI testing from guesswork to data-driven development**

- Actionable insights, not just numbers
- Minutes to instrument, not days
- Fully automated in CI/CD
- Clear quality signals for non-deterministic systems
