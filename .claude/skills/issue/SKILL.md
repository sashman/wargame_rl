---
name: issue
description: Opens a GitHub issue the way this repo requires — jargon-free TL;DR first, the right kind label, needs:prereg on arms, and added to the project board. Use when opening, filing, or tracking any work: a bug found mid-task, a research question, an experiment arm, or a feature.
---

# Opening an issue

⚠ **This skill exists because `gh issue create` bypasses the issue forms entirely.**
`.github/ISSUE_TEMPLATE/*.yml` only runs in the **web UI**. Anything created from
the CLI gets no TL;DR, no required fields and no board placement — which is how
#277, #278 and #281 were opened without any of them.

So the forms are the contract; this skill is how the CLI honours it.

## 1. Check it is not already answered

Cheapest possible step, and the most expensive to skip:

```bash
gh issue list --state all --search "<keywords>"
grep -n "<keywords>" CLAUDE.md reports/README.md
```

`CLAUDE.md` § Settled — do not re-run exists because questions get re-asked. A
refuted hypothesis costs a run to establish and costs another run every time
someone retries it without knowing it failed.

## 2. Pick exactly one kind

| kind | is | title prefix |
|---|---|---|
| `kind:question` | a question the record will answer; owns arms | `Q: ` |
| `kind:arm` | ONE change, ONE comparator, ONE provenance tuple | `arm: ` |
| `kind:bug` | behaviour diverges from `docs/rules/` or its own spec | `fix: ` |
| `kind:build` | feature or refactor; ships behaviour, no verdict | `feat: ` / `refactor: ` |

⚠ **If you are about to write "and" in an arm's title, open two arms.** A bundled
change cannot be attributed. There is deliberately no `experiment` label — an
experiment is a question with N arms.

⚠ **If a build could move a measured number, it is an arm.** Write down why it
needs no verdict; that sentence is where you discover it does.

## 3. Write the TL;DR first, before the body

**Two or three sentences. No jargon. Then a `---`, then the technical body.**

Write it *first*, not last — once you are deep in the provenance table it becomes
very hard to write plainly.

**The test:** could someone who has never seen this repo read it and know what
this is about and why it matters? If it contains `vp`, `K=3`, `paired`,
`refereed`, `decode`, `arm`, `seed base`, `t=`, or a config filename, it fails.

Numbers are fine — *unexplained* numbers are not. "scores +61 against a weak
opponent and +21 against a strong one" is good; "+61.4 vs +20.8 refereed at K=3"
is not.

Worked example (#274):

> If you only ever train against one fixed opponent, you learn to beat *that
> opponent* — not to play well. We have proof: the same agent scores **+61
> against a weak opponent and +21 against a strong one**, while being *further
> behind* the hand-written scripts in the first case. So the score measures the
> opponent, not the agent.

## 4. Fill the fields the form would have required

Read the matching `.github/ISSUE_TEMPLATE/<kind>.yml` and answer its required
fields in the body. Each one prevents a failure this repo has already paid for.

For `kind:arm` the load-bearing ones are: **parent question**, **comparator named
before measuring**, **pre-registration path**, **the criterion with its MDE at
80% power**, seeds, n, refereed config, decode `K`, epochs.

For `kind:bug`: **does the fix void numbers already on file?** If yes, closing it
requires a bullet in `CLAUDE.md` § What voids a number.

## 5. Create it, with labels

```bash
gh issue create --title "<prefix><title>" --body-file <file> \
  --label "kind:<kind>" [--label no-gpu] [--label hold] [--label voids-numbers]
```

- **`no-gpu`** if it runs on the CPU box — that label is the CPU box's work queue.
- **`hold`** if it must not start yet; say why in the body.
- **`needs:prereg` is added automatically to arms by the form, but NOT by the
  CLI.** Add it yourself unless the pre-registration is already committed:

```bash
gh issue create ... --label "kind:arm" --label "needs:prereg"
```

## 6. Add it to the board — the CLI does not

```bash
gh project item-add 2 --owner sashman \
  --url https://github.com/sashman/wargame_rl/issues/<n>
```

New items land with no status; set one:

```bash
gh project item-list 2 --owner sashman --format json \
  --jq '.items[] | select(.content.number==<n>) | .id'

gh project item-edit --id <item-id> \
  --project-id PVT_kwHOAA3SAs4BinKn \
  --field-id PVTSSF_lAHOAA3SAs4BinKnzhhei4A \
  --single-select-option-id <option>
```

| column | option id | when |
|---|---|---|
| Backlog | `dd84daff` | opened, not started |
| Blocked | `8c3dd01e` | waiting on the GPU box or another issue (pair with `hold`) |
| In progress | `58c97fc2` | being worked on now |
| Measured, not landed | `d35e225a` | the run finished, the finding has not reached all four destinations |
| Done | `4989c0c5` | closed, finding landed |

⚠ Projects needs the `project` token scope. If a call fails with
`missing required scopes`, ask the user to run
`gh auth refresh -h github.com -s project` — it is an interactive device flow and
cannot be done for them.

## 7. Wire it to its parent

An arm names its parent question in the body, **and** the parent lists it in a
task list — the relation is recorded from both ends, which is what survives
someone editing one side. This is the only parent/child mechanism in use; there
is no epic label and no Projects hierarchy.

## When Claude opens one unprompted

On finding a defect mid-task: **open the issue immediately** with the evidence
while it is fresh — including whether it voids past numbers — then tell the user
and let them set priority. Do not silently fix scope-creep, and do not lose the
finding to the end of a session.

## Closing

Follow `CLAUDE.md` § Where a finding lands — all four destinations, or the
finding is not landed. Then set the `outcome:` label, remove `needs:writeup`, and
move the board item to Done.

⚠ **`outcome:` labels belong on CLOSED issues.** An open issue carrying one is
either mislabelled or should have been closed.

⚠ **Never reopen a closed issue to revise a finding.** Open a new one saying
`Retracts #N`, add `retracted` to the old one, and comment. Reopening destroys
the close date, which is the timestamp ordering the finding before its retraction.
