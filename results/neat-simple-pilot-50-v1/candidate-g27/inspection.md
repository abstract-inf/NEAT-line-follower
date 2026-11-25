# Exploratory generation-27 inspection

Post-hoc inspection of a non-selected related candidate on consulted data; no new simulation.

| Group | Courses | Candidate / prototype RMSE mm | Candidate / prototype time s |
|---|---:|---:|---:|
| mixed | 30 | 21.985 / 16.345 | 5.7500 / 5.6322 |
| speed | 20 | 5.571 / 4.574 | 7.9658 / 8.1842 |
| corner | 10 | 46.322 / 40.590 | 2.6333 / 2.6333 |

Across 60 shared finishes, time improves only 0.23%, while mean geometric RMSE increases 24.95%.

The candidate is faster on the speed-course group, slower on mixed courses, and has the same mean corner completion time. Its higher tracking error appears in all three groups; episode summaries do not establish where within an individual course the error occurs. Localizing overshoot or corner cutting would require telemetry or replay. Near-cap straight speed alone is insufficient evidence that this is a better controller overall.

Both models finish 60/60 primary courses. Candidate retention is 20/20. Against PID the candidate has 59 shared finishes; the full paired metrics, physical straight measurements, five largest error increases and exact model/source hashes are in inspection.json. This inspection does not change model selection or replace the accepted prototype.

Reproduce without evolution: `& .\.venv\Scripts\python.exe -m research.inspect_simple_candidate`.

Captured pilot snapshot: `5c4e5e5ea6f1f2461040cdaba557279f983f78da5daa11f47a382f5f373a673a`.
