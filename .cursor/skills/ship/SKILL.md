---
name: ship
description: Creates a feature branch, commits staged changes, pushes, and opens a PR. Use when the user says "ship", "branch commit push pr", or wants to create a PR from current changes.
---

# Ship (branch, commit, push, PR)

When the user invokes this skill, run the full ship workflow: **branch only from main** → commit → push → open PR.

**Critical:** New branches must always be created from `main`. Never create a feature branch from another branch.

## Steps

1. **Get branch name and commit message**
   - If the user provided them (e.g. `/ship feature/my-feature Add reward shaping`), use those.
   - Otherwise ask: "Branch name (e.g. feature/my-feature)?" and "Commit message (imperative, e.g. Add reward shaping)?"

2. **Run the ship**
   - In the project root, run:
     ```bash
     just ship <branch> "<commit message>"
     ```
   - This checks out `main`, pulls latest, creates the new branch from `main`, stages all changes, commits with the message as both title and body, pushes, and runs `gh pr create --fill` so the PR title and description are populated from the commit.
   - If you're not on `main`, the recipe runs `git checkout main` and `git pull` first so the new branch is always created from up-to-date `main`.

3. **If something fails**
   - If pre-commit or commit fails, suggest running `just validate` first and fixing any issues, then try again.
   - If the user has nothing staged and doesn’t want to commit everything, suggest staging with `git add <paths>` then running `just ship` again (note: `just ship` runs `git add -A`, so they may prefer to run the git/gh steps manually for partial commits).

## Branch naming (reminder)

Use `feature/<topic>`, `fix/<topic>`, or `refactor/<topic>` — never commit to `main`.
