---
name: release
description: Cuts a GitHub release the way this repo does it — a headline table with its provenance, checkpoints stripped to weights and verified bit-identical, a re-scored reproduction check, SHA256SUMS, and reproduction commands. Use when asked to cut, publish, or tag a release.
---

# Cutting a release

A release here is **a claim with its evidence attached**, not a tarball. The
weights are published so someone can check the table, so the table and the files
have to agree — and be *shown* to agree.

⚠ **A release is the most quotable thing this project produces.** Every number
in it will be repeated without its caveats, so it must not carry any figure the
current measurement standard cannot support. Read `CLAUDE.md` § How to measure
here before writing a single row.

## 1. Decide what the release actually claims

Write the one-sentence claim first. If you cannot write it without a hedge, the
release is premature — say so rather than shipping a hedged headline.

⚠ **"No result" is a legitimate release** when the machinery shipped and the
measurement is honest. `v4.0` is one: the melee stack works, and the headline is
that nothing resolves. That is more useful than a flattering table, and far more
useful than silence.

## 2. Pick the version

Look at what the previous tag claimed, and bump on the size of the change in
*what the project can do or say*, not on lines of code:

```bash
gh release list --limit 5
gh release view <previous> --json body --jq .body | head -40
```

Match its register. The house style is: short declarative headline, a table with
its provenance stated in prose above it, then sections that *narrow* — what the
trait is, what the caveat is, what it is not.

## 3. Gather the table, with provenance in the prose

Every release table states, above it: how many seeds, n per cell, which seed
base, which configs and whether refereed, the decode settings, and that the
comparator was re-measured on the same scenarios. A table without that is not
comparable to anything.

⚠ **State the estimator.** Give the paired per-scenario `t`, not an across-seed
SE — see `CLAUDE.md`. A release quoting the wrong one is the worst place for it.

## 4. Prepare the checkpoints

```bash
just prepare-release dist <label> <ckpt>...
```

`scripts/prepare_release_checkpoints.py` keeps only `state_dict`,
`hyper_parameters`, `hparams_name`, `epoch` and `global_step`; drops optimiser
moments, LR-scheduler, loop and callback state; and drops the `_kl_reference.*`
tensors an anchored run carries. **It verifies every retained tensor with
`torch.equal` and raises if any differs**, then writes `SHA256SUMS.txt`.

⚠ **The reference tensors are the warm start, not the trained policy.** Shipping
them invites someone to score the wrong weights. On this repo's runs stripping
takes 198 MB → 50 MB, half of which is that second copy.

## 5. Re-score one stripped file — this step is not optional

Score a released checkpoint through the *published* command and check it
reproduces its row **to the digit**:

```bash
just measure-melee-ladder dist/<label>-s1-epoch<N>.ckpt
```

⚠ `torch.compile` prefixes every `state_dict` key with `_orig_mod.`, and warm
starts load with `strict=False` — so a mangled checkpoint loads as **nothing at
all** and scores a random network as a trained one. Bit-identity of tensors does
not prove the file *loads*; only re-scoring does.

## 6. Publish

```bash
gh release create <tag> --title "<tag> — <headline>" --notes-file <file> \
  dist/*.ckpt dist/SHA256SUMS.txt
```

Tag the commit the numbers were measured at. Link reports by **tag**, not
`main`, so the links keep working when the files move:
`https://github.com/<owner>/<repo>/blob/<tag>/reports/<file>.md`.

## 7. Sections the notes must carry

- **The claim**, with its table and provenance.
- **What explains it** — one trait, not a list.
- **What this is not.** Name the strongest objection to your own headline. `v3.0`
  says its opponents are hand-written heuristics and there is no self-play;
  `v4.0` says nothing resolves and the play-time helpers do most of the work.
- **Checkpoints** — how many seeds and *why* (a per-seed band, so one file is a
  sample, not "the model"), what was stripped, and the reproduction check.
- **Reproducing** — exact commands, including how to re-measure the comparator.
- **Reports** — links at the tag.
- **Validation** — test count, lint and mypy clean, at that commit.

## 8. After publishing

Note the release in `reports/README.md` if it carries a finding not already
indexed, and close any issue whose finding it lands. A release is a fifth
destination, never a substitute for the four in `CLAUDE.md`.
