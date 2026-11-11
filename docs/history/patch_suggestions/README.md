# Suggested peer patches

These have not been applied to teammate branches or the product's pinned copies.

- naila-stable-cutoff.patch: global score/index ordering at a top-K tie cutoff. Original 13 tests pass but miss this case. A scratch patched run plus the added regression passes 14/14.
- yuyue-windows-provenance.patch: POSIX provenance keys on Windows. Original suite 97/98; scratch patched suite 98/98.

Owners should inspect these on their own branches, then use git apply --check from the repository root before applying. Their target files intentionally are not installed in Kireeti's top-level teammate folders. Revisions, probes and reproduction commands are in ../TEAM_QC_REPORT.md.
