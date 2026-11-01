# How to use these files with Claude Code (8 phases, day by day)

1. Everything in this folder is already in place: CLAUDE.md, docs/, specs/, requirements.txt, scripts/, .claude/settings.json, the datasets in data/raw/.
2. Before Claude Code: plug in the laptop, set Windows Sleep to Never, then start `claude` in this folder (Major Project).
3. Day 1 message (Phase 0 setup + Phase 1): see the long prompt you were given; in short it says:

```
Phase 0: run scripts\setup.ps1 and scripts\preflight.ps1, fix failures, git init, verify both datasets. Then read CLAUDE.md, docs/, specs/01-09.
Then do Phase 1 only (specs/07_phased_build_plan.md): env check, data prep + checks + tests, leakage "before" audit.
Smoke first, then full. Print the PHASE GATE report, commit, update artifacts/RUN_LOG.md, and STOP. Do not start Phase 2.
```

4. Each following day, open a terminal in this folder, run `claude`, and send:

```
Read CLAUDE.md, artifacts/RUN_LOG.md and git log, tell me which phase is current, then do the next phase only per specs/07_phased_build_plan.md.
```

5. To run everything without stopping, send: "run all phases autonomously, stop only on hard stops".
6. After each phase, paste the PHASE GATE report to me to sanity-check, especially after Phase 1 (leakage AUC), Phase 2 (A2 vs A0 calibration) and Phase 5 (Pareto frontier).
