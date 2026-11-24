# Five proposed long-course case studies

Prepared 21 November 2026. **The user selected option 4, Rhythm Changes, from the layouts before controller testing. Its 87-episode comparison is complete.** See [results, frozen controllers, figures and replay commands](../research/Rhythm_Changes_Long_Course_Results.md). The original proposal manifest and previews remain historical records from before testing; the other four courses remain unevaluated.

The runnable track specifications, PNG/SVG maps, overview, section boundaries, geometry hashes, and validation records are under `runs/long-course-candidates-20261121/`. These are open start-to-finish courses: green marks the start, red marks the finish, and arrows show travel direction. Orange dots identify sharp turns. The individual maps include numbered key sections and a distance-based feature strip.

## Options

| ID | Course | Length | Sharp turns | Turnarounds | Main purpose |
| --- | --- | ---: | ---: | ---: | --- |
| **1** | **Balanced Grand Prix** | **34.78 m** | **10** | 2 | Recommended overall comparison: long straights, broad and tighter curves, a six-turn chicane, right-angle steps, and re-acceleration. |
| 2 | High-speed Sweep | 36.50 m | 4 | 2 | The speed-focused option: 6 m launch, broad curves, wide turnarounds, a late sharp chicane, and an almost 6 m finishing straight. |
| 3 | Technical Switchbacks | 38.01 m | 26 | 3 | Dense steering challenges: short-spaced zigzags, right-angle steps, tighter turnarounds, and a 0.22 m-radius S near the end. |
| 4 | Rhythm Changes | 45.77 m | 20 | 3 | Repeated transitions between straights, smooth curves, and sharp chicanes, with a denser late combination. |
| 5 | Recovery Endurance | 47.16 m | 14 | 3 | Three short paint gaps, two crossing distractors, a side spur, mixed bends, and a late corner sequence. |

Sharp-turn counts include zero-radius 45° and 90° vertices; smooth circular turnarounds are counted separately. Difficulty descriptions characterize geometry, not observed controller outcomes.

**Recommendation:** choose **1** for the primary competition-style illustration. It covers the speed-versus-technical-driving question without adding line-loss features. Choose **4** if a longer series of transitions is the main interest. Course **5** emphasizes recovery behavior, which is a different emphasis from pure speed/steering performance.

## What is common across the five

- Paint width is **20 mm**; floor sampling is **2 mm**. Route geometry uses the existing analytic straight/arc/corner primitives and 2 mm route tessellation.
- Arena squares are respectively **12, 13, 9, 11, and 12 m** per side. The raster limits remain within the existing 8192-pixel budget; no robot or sensor configuration is changed.
- Every course has a launch straight, both turn directions, a late technical section, and a finishing straight. All are continuous painted lines except the three explicitly declared gaps in option 5.
- Option 5 gaps are **20, 35, and 45 mm** long. At its two crossings and single side spur, the intended route continues straight; no hidden turn instruction is required. Magenta rings mark gaps and purple diamonds mark distractor junctions in the maps. These schematic marker sizes exaggerate the small features for readability.

The proposal used a primary cap of **1.00 m/s** with a common **180-second time limit**, retaining the existing 100 mm corridor, two-second no-sensor-contact departure rule, and five-second route-progress stall rule. The committed case-study protocol retains these primary rules and scales the timeout below 1 m/s as `180 / min(cap, 1.0)`. It freezes 31 NEAT genomes and five saved PID configurations before testing; NEAT phase selections use the existing validation rankings. Native-cap results and the shared-cap transfer comparison are reported separately. See [the frozen protocol](../runs/rhythm-changes-case-study-20261121/protocol.json).

## Geometry and paint verification

All five passed checks for arena bounds, minimum curve radius, nonadjacent route clearance, finish runout, gap and junction placement, rasterized centerline paint, white gap centers, and JSON roundtrip identity. Canonical geometry hashes are distinct from each other and protected prior experiments, including translated, rotated, or mirrored duplicates. These checks establish geometry validity; controller performance on option 4 is documented separately.

The route-clearance check uses the existing validator's 0.45 m arclength exclusion and 0.20 m minimum spatial clearance, computed in small array chunks to avoid a large all-pairs memory allocation. The closest nonadjacent clearances are approximately **0.277, 0.327, 0.240, 0.327, and 0.327 m**. Raster checks run one course at a time. The course suite registers the five reserved geometries with the existing protected-track scan for future experiments.

## Files and commands

Overview: [five-course-overview.png](../runs/long-course-candidates-20261121/five-course-overview.png), [SVG](../runs/long-course-candidates-20261121/five-course-overview.svg).

Individual maps and runnable tracks:

| ID | Preview | Track JSON |
| --- | --- | --- |
| 1 | [Balanced Grand Prix](../runs/long-course-candidates-20261121/01-balanced-grand-prix.png) | [Track](../runs/long-course-candidates-20261121/01-balanced-grand-prix.json) |
| 2 | [High-speed Sweep](../runs/long-course-candidates-20261121/02-high-speed-sweep.png) | [Track](../runs/long-course-candidates-20261121/02-high-speed-sweep.json) |
| 3 | [Technical Switchbacks](../runs/long-course-candidates-20261121/03-technical-switchbacks.png) | [Track](../runs/long-course-candidates-20261121/03-technical-switchbacks.json) |
| 4 | [Rhythm Changes](../runs/long-course-candidates-20261121/04-rhythm-changes.png) | [Track](../runs/long-course-candidates-20261121/04-rhythm-changes.json) |
| 5 | [Recovery Endurance](../runs/long-course-candidates-20261121/05-recovery-endurance.png) | [Track](../runs/long-course-candidates-20261121/05-recovery-endurance.json) |

[Manifest and sector boundaries](../runs/long-course-candidates-20261121/manifest.json); [protected course suite](../runs/long-course-candidates-20261121/suite.json).

```powershell
$py = ".\.venv\Scripts\python.exe"

# Recheck the five layouts without writing files or running any controllers
& $py prepare_long_course_candidates.py --check-only

# Recreate the same track files and PNG/SVG previews; still no controller tests
& $py prepare_long_course_candidates.py
```

Changed geometry will not silently replace an existing published track file. Use a new output directory for revisions. This helper has no evolution or controller-assessment command. After the course is chosen, the later case-study results should be stored separately from these untouched proposal files and interpreted alongside the [five-seed shared-track comparison](../research/NEAT_Fast_Five_Seed_Results.md).
