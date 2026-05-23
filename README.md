# From Legacy to Modern: A Framework for Diagnosing and Modernizing Data Pipelines for the Data-Driven Workforce

**Lisette M. Kamper-Hinson | M.S. Computer Science, Wake Forest University**
**Advisor: Dr. William H. Turkett | Committee: Dr. Ron P. Doyle, Dr. Jeffrey D. Camm**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io/)
[![Tableau](https://img.shields.io/badge/Tableau-Visualization-orange)](https://www.tableau.com/)

---

## Overview

Modernization of operational data pipelines frequently emphasizes analytical sophistication while neglecting structural governance, reproducibility, and decision usability. In manufacturing environments, this imbalance exacerbates misalignment between reported and realized production capacity.

This thesis presents a structured, six-stage framework for modernizing legacy capacity planning pipelines while preserving embedded domain knowledge and minimizing technical debt. The framework is validated through a real-world single-plant manufacturing case study using read-only operational data.

**Result: Capacity utilization improved from 70% to 90% through structured pipeline modernization.**

## Thesis Poster
___

![Thesis Poster](Kamper-Hinson_Thesis_Poster.jpg)

---

## The Problem

Organizations face a critical paradox: they invest in modern cloud infrastructure while the internal analytical pipelines responsible for extraction, transformation, modeling, and reporting remain fragile and opaque. These pipelines, built incrementally over years, embed domain heuristics, undocumented transformations, and tightly coupled logic that resists adaptation.

Existing frameworks like CRISP-DM govern building new pipelines but offer no guidance for inheriting and transforming existing ones. No integrated, experimentally validated modernization methodology existed before this work.

---

## The Six-Stage Framework

### Phase 1: Understanding

**Stage 1: Domain Alignment**
Establish pipeline context, organizational goals, end users, constraints, and the data environment. Domain expert meetings are used to capture embedded operational knowledge before any technical changes are made.

**Stage 2: Structural Audit and Decomposition**
Classify components as preserve, refactor, or redesign. Evaluate impact, risk, maintainability, and alignment. Produce a modernization roadmap. Domain expert validation is required before advancing.

**Stage 3: Functional Evaluation and Prioritization**
Evaluate candidate techniques, audit data, compare baselines, and assess interpretability and technical debt. Applied per pipeline component, not globally.

### Phase 2: Analysis and Design

**Stage 4: Governed Modeling Introduction**
Evaluate candidate modeling techniques, compare baselines, and assess interpretability and technical debt. Applied per pipeline component. Models may be revisited multiple times across the pipeline.

**Stage 5: Visualization and Communication Design**
Anchor each visual to a decision context. Preserve operational knowledge and model uncertainty. Applied per decision during output. May be revisited as new outputs are exposed or analytical behavior changes.

### Phase 3: Implementation

**Stage 6: Standardize and Validate**
Reproducible execution, system health view, validate analytical, communicative, and operational intent.

---

## Mathematical Foundations

### Capacity Fill Rate (CFR)

The core performance metric computed at each stage:

```
CFR_d = Realized Output_d / Effective Hours_d
```

Effective capacity accounts for theoretical maximum minus downtime, maintenance, and partial-day interruptions, using domain-expert-validated heuristics preserved during modernization.

### Holt's Linear Exponential Smoothing

Used for short-horizon capacity forecasting, decomposing the time series into level and trend components:

```
L_t = alpha * C_t + (1 - alpha) * (L_{t-1} + T_{t-1})
T_t = beta * (L_t - L_{t-1}) + (1 - beta) * T_{t-1}
Forecast: C_{t+h} = L_t + sum(phi^i * T_t) for i=1 to h
```

### Newsvendor Model: Optimal Capacity Buffer

Used in the scenario simulation dashboard to set statistically optimal capacity buffer thresholds, balancing the cost of under-producing against the cost of idle capacity:

```
C(q, D) = c_over * max(0, q - D) + c_under * max(0, D - q)

Optimal planned capacity:
q* = F^{-1}(c_over / (c_over + c_under))
```

Where Cu = underage cost (lost capacity) and Co = overage cost (idle capacity).

---

## Streamlit Dashboard Suite

The framework was delivered as a four-dashboard Streamlit application for daily use by operations managers and senior leadership.

| Dashboard | Purpose |
|---|---|
| 01 Capacity Overview | Daily operations view showing planned output vs. realistic achievable capacity with weekly load indicators |
| 02 Sales and Commitments View | Answers what can be safely promised to customers, broken down by container type (Quart, Gallon, Pail) with low, typical, and high ranges |
| 03 Efficiency Gap Analysis | Shows where capacity is being lost through downtime, changeovers, or run rate, and what closing the gap to 90% efficiency means in real production units |
| 04 What-If View | Allows custom scenario modeling by adjusting shift schedule, downtime, changeovers, and efficiency assumptions |

---

## Key Results

- Capacity utilization improved from 70% to 90% through structured pipeline modernization
- Framework validated through a real single-plant manufacturing case study with read-only operational data
- All six stages applied sequentially with domain expert validation at each gate
- Delivered as a fully documented, production-ready analytical system with SOPs for ongoing maintenance

---

## Contribution to the Field

This thesis fills a documented gap in data science methodology. CRISP-DM and similar frameworks govern building new pipelines from scratch but provide no guidance for inheriting and transforming existing ones. This work provides the first prescriptive, experimentally validated framework specifically designed for legacy pipeline modernization, treating modeling, visualization, and implementation standards as integrated system design decisions rather than independent technical choices.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | ETL pipeline, statistical modeling, forecasting |
| Streamlit | Four-dashboard interactive application suite |
| Tableau | Executive-facing KPI dashboards |
| Holt's Exponential Smoothing | Short-horizon capacity forecasting |
| Newsvendor Model | Stochastic capacity buffer optimization |
| GitHub | Version control and reproducibility |

---

## Author

**Lisette M. Kamper-Hinson**
M.S. Computer Science, Wake Forest University
[LinkedIn](https://www.linkedin.com/in/lisette-kamper-hinson) | [GitHub](https://github.com/LisetteKH)
lisette.hinson@gmail.com
