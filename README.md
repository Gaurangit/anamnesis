# anamnesis

Narrative-query and procedural-knowledge layer over the knowledge artifacts stack.
(full workflow is not updated yet, dependecy and related repos havn't updated, will update soon)

## What it does

1. **Narrative query.** Take a free-form essay, decompose it into a structured
   retrieval intent, plan and execute against the KO registry (vector + SPARQL +
   `kg_bridge`), synthesize a lens-shaped result with full provenance, and run
   an adversarial critic that searches the inverse-polarity neighbourhood for
   evidence that complicates the synthesis.
2. **Procedural-knowledge domain.** Adds a PKO-aligned domain under
   [`knowledge_objects/domains/procedural/`](../knowledge_objects/domains/procedural/)
   so the same narrative-query mechanism can surface decision-trace precedent
   alongside declarative knowledge.

## What it is not

* Not an extraction pipeline. Anamnesis only **reads** the KO registry.
* Not a "context graph." That phrase has marketing baggage; the system is
  "narrative query over Knowledge Objects" plus "procedural-knowledge domain
  extension."
* Not an agent framework. The pipeline has fixed stages with defined contracts.



## Architecture decisions
**will be uploded**
