# Cost Analysis — Final AgriRAG Baseline

**Pricing verification date:** 2026-10-01  
**Final generator:** Cohere `command-r7b-12-2024`

The course requires a per-query cost breakdown, 1K / 10K / 100K monthly-query scenarios, competitor comparison, and suggested client pricing. The final AgriRAG architecture keeps retrieval, embeddings, and reranking local, so the only per-query API charge is answer generation.

## 1. Cost architecture

| Component | Implementation | API cost |
|---|---|---:|
| Document embeddings | `intfloat/multilingual-e5-small`, local | $0 |
| Query embeddings | same local E5 model | $0 |
| Vector search | local Chroma | $0 |
| BM25 + RRF | local | $0 |
| Reranking | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, local | $0 |
| Generation | Cohere `command-r7b-12-2024` | variable |
| RAGAS benchmark | Cohere Trial key | $0 for the completed trial run |

Cohere lists Command R7B production pricing at **$0.0375 / 1M input tokens** and **$0.15 / 1M output tokens**.

Official model page: https://docs.cohere.com/docs/command-r7b  
Official pricing explanation: https://docs.cohere.com/docs/how-does-cohere-pricing-work

## 2. Per-query planning envelope

Run #21 did not persist Cohere `billed_units`, so the following token counts are explicitly a **conservative planning assumption, not a measured billing claim**.

The final pipeline sends at most five reranked chunks to the generator. With a target chunk size of about 350 tokens, the evidence budget is at most about 1,750 tokens before prompt/source metadata. The completed 20-question RAGAS run produced answers averaging about **697 characters**. For budgeting we therefore use:

- Input: **2,500 tokens/query**
- Output: **250 tokens/query**

Production model cost:

```text
Input  = 2,500 / 1,000,000 × $0.0375 = $0.00009375
Output =   250 / 1,000,000 × $0.15   = $0.00003750
Total variable LLM cost/query        = $0.00013125
```

Because embedding and reranking are local, the planned **API variable cost/query is $0.00013125**.

## 3. Required monthly query scenarios

| Monthly queries | Generation cost | Embedding API | Rerank API | Total variable API cost |
|---:|---:|---:|---:|---:|
| 1,000 | $0.13 | $0 | $0 | **$0.13** |
| 10,000 | $1.31 | $0 | $0 | **$1.31** |
| 100,000 | $13.13 | $0 | $0 | **$13.13** |

For reference, at **1,000,000 queries/month**, the same planning envelope gives about **$131.25/month** in Cohere generation cost.

## 4. User-volume view

If an active user asks **10 questions/month**:

| Active users | Queries/month | Planned generation cost |
|---:|---:|---:|
| 1,000 | 10,000 | **$1.31** |
| 10,000 | 100,000 | **$13.13** |
| 100,000 | 1,000,000 | **$131.25** |

## 5. Hosting

### Railway
Railway currently lists a **Free** plan at $0/month for small experiments and **Hobby** at a $5/month minimum usage commitment. Hobby includes $5 of resource usage; if actual Railway resource usage is below $5, the monthly Railway bill remains $5. AgriRAG loads local embedding and reranker models, so Hobby is the safer planning tier than a 0.5 GB free container.

Official pricing: https://docs.railway.com/pricing/plans

If Railway resource usage stays inside the included $5, a simple planning floor becomes:

| Queries/month | Cohere | Railway minimum | Planning floor |
|---:|---:|---:|---:|
| 1,000 | $0.13 | $5.00 | **$5.13/month** |
| 10,000 | $1.31 | $5.00 | **$6.31/month** |
| 100,000 | $13.13 | $5.00 | **$18.13/month** |

Actual Railway usage can exceed the included amount and must be read from the live deployment invoice/usage dashboard.

## 6. Competitor comparison

The course appendix asks for comparison with Mendeley and Zotero.

| Product | Current public pricing / AI position | Comparison with AgriRAG |
|---|---|---|
| Mendeley | Free plan includes 5 Reading Assistant questions. Plus is $4.99/month, Pro $9.99/month, Max $14.99/month. Pro/Max include Ask My Library. | Mature reference manager with integrated research AI. AgriRAG is narrower: agriculture-only, custom corpus, explicit hybrid retrieval, local reranking, and reproducible RAG evaluation. |
| Zotero | Zotero is a free research/reference manager; optional individual file storage is 300 MB free, 2 GB $20/year, 6 GB $60/year, or unlimited $120/year. | Zotero is primarily a reference-management and research-organization product, so its public storage pricing is not directly comparable to AgriRAG's per-query LLM inference cost. |

Official sources:
- Mendeley pricing: https://www.mendeley.com/pricing/
- Zotero storage: https://www.zotero.org/storage/
- Zotero overview: https://www.zotero.org/support/quick_start_guide

## 7. Suggested client pricing

These are **commercial planning suggestions**, not observed market prices:

| Offer | Suggested price | Included usage |
|---|---:|---|
| Public portfolio/demo | Free | heavily rate-limited |
| Individual agriculture learner | $4.99/month | up to 500 questions |
| Agronomist / professional | $9.99/month | up to 2,000 questions |
| Small team pilot | $49/month | up to 10,000 questions |

At the current Command R7B token rates, model inference is a small part of these prices. The commercial margin primarily has to cover hosting, support, monitoring, source maintenance, user management, taxes/payment fees, and future model/provider changes.

## 8. Important limitations

- The completed benchmark used a free Cohere Trial key, which is not a production SLA.
- Trial usage is rate-limited; run #21 was deliberately paced below the 20 requests/minute trial limit.
- Exact production cost should be recalculated from Cohere's returned `billed_units` once the live app has representative traffic.
- Agriculture documents and model pricing can change; both source corpus and prices should be versioned before a commercial launch.
