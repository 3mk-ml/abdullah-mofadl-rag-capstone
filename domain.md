# Domain Definition — AgriRAG

## Domain
Evidence-grounded agricultural knowledge for **irrigation, soil fertility, plant nutrition, crop water stress, water harvesting, and climate-smart agriculture in arid and semi-arid environments**, with special relevance to Yemen and the wider West Asia and North Africa region.

## Problem
Farmers, agricultural students, extension workers, and agritech teams often need answers that are scattered across long technical manuals and research reports. A general-purpose LLM can answer fluently but may invent agronomic thresholds, fertilizer doses, irrigation quantities, or citations. AgriRAG retrieves evidence from a curated technical corpus and requires the answer to be grounded in those sources.

## Target users
- Agricultural students and researchers
- Extension workers and agronomists
- Agritech developers working on irrigation/soil/crop monitoring
- Farmers or field teams who need an understandable explanation of official guidance

## In-scope questions
- Crop water requirements and evapotranspiration concepts
- Irrigation scheduling and irrigation methods
- Deficit and supplemental irrigation
- Soil-water relationships and soil testing
- Soil fertility and plant nutrient management
- Water quality and salinity considerations
- Water harvesting and conservation agriculture
- Climate-smart agriculture and dryland resilience

## Out of scope
- Diagnosing a specific farm without sufficient measurements
- Inventing crop-specific thresholds not contained in the sources
- Prescribing pesticide, fertilizer, or irrigation doses beyond cited guidance
- Replacing a licensed agronomist or local extension authority

## Source policy
The corpus must contain 20–50 high-quality documents. Preference order:
1. FAO and FAO Open Knowledge publications
2. ICARDA/CGIAR research and technical manuals
3. World Bank technical reports relevant to climate-smart agriculture and Yemen
4. Peer-reviewed or institutionally reviewed manuals with clear provenance

Each indexed chunk stores source filename and page number so the UI can show traceable evidence.

## Language
Primary source language: English. User questions may be English or Arabic. The retrieval layer uses multilingual embeddings and a multilingual reranker so Arabic questions can retrieve English source passages. The interface supports Arabic RTL and uses the Amiri font.

## Success criteria
- 20–50 curated documents in the final corpus
- 30 manually reviewed golden questions
- Recall@5 >= 80%
- Hybrid retrieval (vector + BM25 + RRF) followed by reranking
- 20-question RAGAS report
- Public Streamlit demo with simple authentication
- One-page ADR and cost analysis for 1K, 10K, and 100K monthly queries/users as required by the course
