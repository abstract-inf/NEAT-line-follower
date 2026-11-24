# Related work: PID speed control and line following

Living reference notes for the manuscript. Recorded **19 November 2026**. Prioritize published journal and conference work; preprints are excluded from the core list. Publication alone does not establish methodological quality. These references motivate controller design choices; they do not establish that our exact implementation is superior to conventional PID.

## 1. Dynamic speed control for a line-following robot

**Ardy Seto Priambodo, Aris Nasuha, and Oktaf Agni Dhewa (2024).** *Integrated Implementation of Computer vision and PID Control for Dynamic Speed Control of Line follower Robot.* Telekontran, 12(1). Published 25 July 2024. [Publisher record and abstract](https://ojs.unikom.ac.id/index.php/telekontran/article/view/13323); [DOI: 10.34010/telekontran.v12i1.13323](https://doi.org/10.34010/telekontran.v12i1.13323).

- **Verified evidence:** Publisher metadata and abstract reviewed. The authors combine camera-based line detection and PID control in a simulated e-puck robot in Webots. Their dynamic-base-speed approach reportedly reduces completion time from 58.08 to 50.386 seconds in one arena (13.3%) and from 71.584 to 66.624 seconds in another (6.9%). [Source](https://ojs.unikom.ac.id/index.php/telekontran/article/view/13323)
- **Relevance:** Direct precedent for varying forward speed alongside PID line tracking. It supports discussing speed control as an addition to steering control.
- **Limits:** Different sensors, robot, simulator, and courses. The full method and experimental design still need detailed review before a final manuscript comparison. We have not verified that it uses our reciprocal slowdown equation or remembered-side recovery. These reported improvements apply to its experiments, not ours.

## 2. Practical modifications to PID line following

***Optimization of PID Control for High Speed Line Tracking Robots* (2015).** Procedia Computer Science, 76, 147–154. [Publisher record](https://www.sciencedirect.com/science/article/pii/S1877050915038302); [DOI: 10.1016/j.procs.2015.12.329](https://doi.org/10.1016/j.procs.2015.12.329). Verify complete author metadata before exporting the final bibliography.

- **Verified evidence:** Publisher abstract and metadata reviewed through search; direct full-text retrieval failed. The work tests a prototype with ten infrared reflective sensors and describes a customized combination of closed-loop PID and open-loop control. The publisher explicitly assigns peer-review responsibility to the IRIS 2015 organizing committee. [Source](https://www.sciencedirect.com/science/article/pii/S1877050915038302)
- **Relevance:** Useful context for treating a practical PID baseline as a complete controller with additional engineering logic.
- **Limits:** Detailed algorithms and comparative results remain to be reviewed. Do not cite it as proof of our speed law, recovery scheme, or performance advantage.

## 3. Established PID implementation background

**Karl J. Astrom and Richard M. Murray.** *Feedback Systems: An Introduction for Scientists and Engineers*, online 2008 manuscript, Chapter 10, “PID Control.” **Textbook background, not an experimental line-following paper.** [Author-hosted chapter PDF](https://www.cds.caltech.edu/~murray/books/AM08/pdf/am08-pid_02Dec08.pdf).

- **Verified evidence:** Relevant chapter text reviewed, including derivative filtering, integral windup, and Figure 10.11 showing filtered derivative and anti-windup compensation. The text explains the high-frequency noise sensitivity of ideal differentiation and practical filtering. [Source, especially printed pages 306–308](https://www.cds.caltech.edu/~murray/books/AM08/pdf/am08-pid_02Dec08.pdf)
- **Relevance:** Background for filtering the derivative and protecting the integral state in a practical PID implementation.
- **Limits:** Our bounded conditional integration is a project-specific heuristic, not a reproduction of the chapter's back-calculation design. The chapter does not validate our error-based speed law or recovery logic.

## 4. Combined vision-based steering and speed control

**Rihem Farkh and Khaled Aljaloud (2023).** *Vision Navigation Based PID Control for Line Tracking Robot.* Intelligent Automation & Soft Computing. [Publisher article](https://www.techscience.com/iasc/v35n1/48167/html); [DOI: 10.32604/iasc.2023.027614](https://doi.org/10.32604/iasc.2023.027614).

**Reviewed:** Publisher abstract and introduction. The authors describe camera-based navigation with cascade PID steering and speed control on a real prototype. This provides context for separate speed and steering responsibilities, but is not our sensor-array controller or exact slowdown heuristic. Full comparative methods remain to be reviewed. [Source](https://www.techscience.com/iasc/v35n1/48167/html)

## 5. Neural line following compared with PID

**Hugo M. Leal, Ramiro S. Barbosa, and Isabel S. Jesus (2025).** *Control of a Mobile Line-Following Robot Using Neural Networks.* Algorithms, 18(1), 51. [Publisher article](https://www.mdpi.com/1999-4893/18/1/51); [DOI: 10.3390/a18010051](https://doi.org/10.3390/a18010051).

**Reviewed:** Publisher abstract. An infrared-sensor experiment uses PID-generated data to train an LSTM; another uses camera inputs and a CNN. The authors report PID as simpler and more efficient on their tested tracks. This is useful counterevidence to universal neural superiority. It is not NEAT, and the camera experiment does not match our input conditions. Read full methods before numerical comparisons. [Source](https://www.mdpi.com/1999-4893/18/1/51)

## 6. Original NEAT algorithm

**Kenneth O. Stanley and Risto Miikkulainen (2002).** *Evolving Neural Networks through Augmenting Topologies.* Evolutionary Computation, 10(2), 99–127. [Author-hosted final paper](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf); [DOI: 10.1162/106365602320169811](https://doi.org/10.1162/106365602320169811).

**Reviewed:** Author-hosted journal-paper metadata and abstract through search; full PDF retrieval timed out. The paper introduces evolution of network topology alongside weights. Cite it for the underlying method, not as evidence of superiority to PID on our task. Full methods still need detailed review. [Source](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)

## How these sources relate to our controller

Our baseline is **filtered PID with error-based speed control and line-loss recovery**. Its normalized forward command is:

```text
forward = max(minimum_command, base / (1 + slowdown_gain * abs(sensor_error)))
```

This equation and the recovery settings are engineering choices implemented and tuned for this project. The sources above provide related ideas and practical PID background; none has been verified as an exact source for this complete implementation. We should neither claim we invented the general idea of slowing for tracking error nor imply that we reproduced a published controller.

The exact local algorithm is documented in [PID controller methodology](PID_Controller_Methodology.md), including sensor error, wheel mixing, conditional integration, and recovery. Our current comparison is recorded in [NEAT generation 134 versus full-speed PID](NEAT134_vs_Fullspeed_PID_100_Results.md). That comparison evaluates complete frozen controllers; it does not isolate the benefit of the PID additions.

## Draft wording for the related-work section

Published line-following studies have examined dynamic base speed alongside PID tracking (reference 1), customized closed-loop and open-loop control (reference 2), and separate steering and speed regulation using vision (reference 4). Neural controllers have also been compared with PID, with results that do not uniformly favor neural control (reference 5). Our study applies NEAT (reference 6) and compares its resulting controllers against a tuned PID-based baseline whose additional speed and recovery rules are explicitly specified.

## Before making stronger claims in the paper

- Read the full methods and results of all research-paper entries before finalizing comparisons; abstracts alone cannot establish equivalent implementations or general robustness. Complete citation metadata before manuscript submission.
- To measure the benefit of our speed rule, compare separately tuned fixed-base and variable-base PID controllers on the same fresh tracks. A second ablation would be needed to isolate recovery behavior.
- Freeze tuning and model selection before loading final test tracks. Report actual chassis speeds, completion, tracking error, jitter, and paired successful-course times.
- Evaluate independently evolved NEAT seeds before making claims about training reliability. A selected generation-134 controller alone is a case study, not evidence of superiority of an entire algorithm family.

