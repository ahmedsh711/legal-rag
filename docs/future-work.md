# Future work

Known gaps and next steps, roughly in priority order. Each item says why it matters and what "done" looks like. Numbers come from the measurements described in the README.

## High priority

### A bigger, independent evaluation set
The golden set has 28 Arabic/English question pairs written by the author, and the answerability gate threshold was tuned on it. Good for catching regressions, too small and too close to the system to claim accuracy.
- Collect 150+ questions written by people who did not build the system, including real questions from the API logs (PII is already redacted there).
- Keep a held-out split that is never used for tuning.
- Done when hit@1, citation recall and refusal rates are reported with confidence intervals on the held-out split.

### An independent judge with negative human labels
Faithfulness is scored by a judge from the same model family as the generator, and all 20 human-labelled answers were "supported", so calibration relies on synthetic negatives.
- Use a judge from another model family.
- Label at least 20 real unsupported answers (the small vLLM model produces plenty).
- Run the RAGAS metrics that were skipped for quota reasons: answer relevancy and context precision.
- Then remove the waiver in `data/golden/smoke_waivers.json`: the current judge scores one faithful Arabic answer (`p05-ar`, a near-verbatim restatement of Articles 120 and 121) at 0.4 on every run.

### Make the self-hosted model usable
Qwen2.5-1.5B (AWQ) on the 4 GB GPU states the right facts but cites the right article in only 6 % of answers, so the citation check refuses 90 % of them. It is a fallback today, not a production option.
- Try a larger quantised model where memory allows, or constrain the output format so every sentence ends with an `[Art. N]` tag.
- A LoRA fine-tune on cited answers from the Gemini runs is the next step if prompting is not enough.
- Done when the local model passes the same quality gate as the hosted one.

## Medium priority

### Guardrails
- Obfuscated prompt injections (base64, spaced letters, paraphrases) pass the pattern layer; add a classifier (for example a Jev yes/no question) and measure its false-positive rate on real questions and article texts, as was done for the patterns.
- Names and street addresses are not redacted. Add an Arabic-capable NER model and measure recall on a labelled sample.

### Drift and online evaluation
- The drift reference is 66 events from evaluation runs. Replace it with a window of known-good production traffic (hundreds of events), then re-run the A/A check.
- Schedule the online evaluation job in the Airflow `monitoring` DAG and record its first real result.
- Send alerts to a real channel (Slack or Discord webhook) instead of the local alert sink.

### Continuous training end to end
- The CT workflow on GitHub cannot reach the local data and Qdrant. Either register a self-hosted runner (with fork pull requests requiring approval, since the repository is public) or move the data plane to a cloud environment.
- Set `GITHUB_DISPATCH_TOKEN` so the Airflow drift check can start the workflow.

### Serving capacity
- vLLM runs with `--max-num-seqs 8` while its KV cache was only 15 % used at that cap. Raise it and repeat the load test, watching time to first token.
- Shadow deployments were dropped because a third API container exhausted the laptop's Docker memory. Revisit on hardware with spare capacity; mirrored traffic must not hold the client connection.

## Low priority

### Kubernetes
The manifests are validated but have not been applied. Deploy them on a local cluster (kind or Docker Desktop), add Qdrant and Redis as StatefulSets or managed services, an Ingress with canary weights, and a HorizontalPodAutoscaler based on CPU or request rate.

### Infrastructure as code
- Move Terraform state to a remote backend with locking.
- Manage the rest of the stack (Qdrant, Redis, monitoring) with the same module, or move to managed services.

### Product
- A semantic cache, once real traffic shows how often questions repeat. Measure the repeat rate first.
- A/B tests of prompts or retrieval settings driven by `/feedback` ratings, once there is enough traffic for significance.
- Fix the digit order of article numbers quoted inside Arabic article bodies (a PDF extraction artefact). Citations already use the parsed article numbers.

### Repository
- Choose and add a license.
