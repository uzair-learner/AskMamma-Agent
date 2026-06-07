# Agent Diagram

```mermaid
flowchart TD
    S[SupervisorAgent]
    I[InventoryAgent]
    F[ForecastAgent]
    D[DocumentAgent]
    R[ResearchAgent]
    P[ReportingAgent]
    Q[QualityReviewAgent]
    M[MemoryAgent]

    S --> I
    S --> F
    S --> D
    S --> R
    I --> P
    F --> P
    D --> P
    R --> P
    P --> Q
    Q --> M
```
