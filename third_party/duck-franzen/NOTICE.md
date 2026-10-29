# Third-party code: the Duck harness, Daniel Franzen's Milestone #2 fork

This was vendored verbatim under D-045. Any later change to it is made only in a commit that says so.

- **Upstream:**
  - Source: https://github.com/tonghuikang/daniel-franzen-arc-agi-3 at commit
    `f472820b24f85e4bfe7790c66bce7d6463b5d568` (2026-10-04). It was downloaded
    as a tarball on 2026-10-29.
  - That repository forks https://github.com/da-fr/arc-agi-3-solution
    (Apache-2.0). This is Daniel Franzen's Milestone #2 solution, which placed
    1st at Milestone #2 with a public leaderboard score of 27.89.
  - Franzen's solution builds on Tufa Labs' Duck harness
    (https://github.com/Tufalabs/duck-harness), by Jeroen Cottaar and Tufa
    Labs, which won Milestone #1.
- **Licence:**
  - This code is Apache-2.0; see `LICENSE` in this directory. The rest of this
    repository is MIT-0.
  - Tufa's own `duck-harness` repository has no LICENSE file. The Apache-2.0
    grant used here is the one that Franzen's repositories declare.
  - Whether this licence is acceptable for prize eligibility is still an open
    question, recorded in D-045.
- **Copied:**
  - `ARC3-Inference/`
  - `tufa-arc-agi-framework/`
  - `serving/`
  - `README.md`
  - `WRITEUP.md`
  - `kaggle/COMPARISON.md`
  - `kaggle/SUMMARY.md`
  - `kaggle/dfranzen/`: the submitted notebook, its source bundle and its
    harness patch.
- **Not copied:**
  - `docs/`, which holds images for the write-up.
  - The lordhansolo and sirikilohit notebook copies under `kaggle/`.
  - The Tufa share notebook.
