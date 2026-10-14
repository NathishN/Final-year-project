# Phase 5.2 — Split Integrity & Session Audit Report

- **Total Manifest Samples:** 56036
- **Total Unique Sessions:** 80
- **Cross-Split Sessions in Existing Split V2:** 1

### Analysis of Session 20200730
In the existing split v2:
- **119 Banana Grade 2 images** are assigned to **test** (enabling Grade 2 test evaluation).
- **147 non-Grade-2 images** (15 Apple Grade 0, 85 Banana Grade 0, 47 Banana Grade 3) are assigned to **train**.
While there is zero path overlap and zero image-hash overlap, full session-level independence was technically not achieved because acquisition metadata and illumination from session `20200730` were present in both train and test splits.
