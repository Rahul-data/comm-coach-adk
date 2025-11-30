# Architecture Diagram

```mermaid
graph TB
    A[Video Input] --> B[Sequential Analysis Pipeline]
    B --> C[Vision Agent]
    B --> D[Voice Agent]
    B --> E[Language Agent]
    C --> F[Parallel Coach Agent]
    D --> F
    E --> F
    F --> G[Aggregator Agent]
    F --> H[Recommender Agent]
    G --> I[Feedback Report]
    H --> I
