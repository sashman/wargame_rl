---
name: ship
description: Creates a feature branch, commits staged changes, pushes, and opens a PR. Use when the user says "ship", "branch commit push pr", or wants to create a PR from current changes.
---

# Ship (branch, commit, push, PR)

When the user invokes this skill, run the full ship workflow: **branch only from main** → commit → push → open PR → set PR description from branch changes.

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
   - This stashes the working tree (`git stash -u`), checks out `main`, pulls latest, creates the new branch from `main`, restores the stash, stages all changes, commits with the message as both title and body, pushes, and runs `gh pr create --fill` so the PR title and description are populated from the commit.
   - The checkout-and-pull of `main` happens unconditionally, so the new branch is always created from up-to-date `main`.
   - **If the branch already exists**, the recipe silently checks it out instead of creating it — so the "always from main" guarantee does not hold for a name you have used before. Pick a fresh name for each PR.

3. **Set PR description from branch changes**
   - After the PR is created, generate a short description from the changes in the branch (vs `main`).
   - Run `git log main..HEAD --oneline` and `git diff main --stat` (or `--name-only`) to see commits and changed files.
   - Compose a PR body that includes: (1) a short summary (one or two sentences) of what the PR does, and (2) a "Changes" section listing notable files or areas (e.g. "env: add VP to Battle and BattleView", "mission: new VP calculator and registry").
   - Write the body to a file, then set it via the REST API:
     ```bash
     cat > /tmp/pr-body.md <<'EOF'
     <description>
     EOF
     PR=$(gh pr view --json number --jq .number)
     REPO=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
     gh api -X PATCH "repos/$REPO/pulls/$PR" -F body=@/tmp/pr-body.md --jq '.html_url'
     ```
   - **Do not use `gh pr edit --body`.** On this repo it fails with a deprecated Projects (classic) GraphQL error (`repository.pullRequest.projectCards`) and leaves the body unchanged — the command reports the error but the PR silently keeps the `--fill` body. The REST endpoint above is unaffected.
   - Verify it applied: `gh api "repos/$REPO/pulls/$PR" --jq '.body | length'` should match the body you wrote, not the shorter commit-derived one.
   - Using a heredoc file rather than an inline `--body` string also avoids shell-quoting damage to markdown tables, backticks, and emoji.

4. **If the PR shows `BEHIND`**
   - `main` moved between the branch point and the push. Merge it in and push again:
     ```bash
     git fetch origin && git merge origin/main -m "Merge branch 'main' into <branch>"
     git push
     ```
   - This re-runs CI on the merge commit. Do not rebase a pushed branch.

5. **If something fails**
   - If pre-commit or commit fails, suggest running `just validate` first and fixing any issues, then try again.
   - If the user has nothing staged and doesn't want to commit everything, suggest staging with `git add <paths>` then running `just ship` again (note: `just ship` runs `git add -A`, so they may prefer to run the git/gh steps manually for partial commits).

## Branch naming (reminder)

Use `feature/<topic>`, `fix/<topic>`, or `refactor/<topic>` — never commit to `main`.

## PR titles

Use conventional commits (`feat:`, `fix:`, `refactor:`, `chore:`, `docs:`) followed by a space and a lower case letter — CI enforces this.
