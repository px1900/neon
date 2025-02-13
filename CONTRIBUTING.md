# How to contribute

Howdy! Usual good software engineering practices apply. Write
tests. Write comments. Follow standard Rust coding practices where
possible. Use `cargo fmt` and `cargo clippy` to tidy up formatting.

There are soft spots in the code, which could use cleanup,
refactoring, additional comments, and so forth. Let's try to raise the
bar, and clean things up as we go. Try to leave code in a better shape
than it was before.

## Submitting changes

1. Get at least one +1 on your PR before you push.

   For simple patches, it will only take a minute for someone to review
it.

2. Don't force push small changes after making the PR ready for review.
Doing so will force readers to re-read your entire PR, which will delay
the review process.

3. Always keep the CI green.

   Do not push, if the CI failed on your PR. Even if you think it's not
your patch's fault. Help to fix the root cause if something else has
broken the CI, before pushing.

*Happy Hacking!*

# How to run a CI pipeline on Pull Requests from external contributors
_An instruction for maintainers_

## TL;DR:
- Review the PR
- If and only if it looks **safe** (i.e. it doesn't contain any malicious code which could expose secrets or harm the CI), then:
    - Press the "Approve and run" button in GitHub UI
    - Add the `approved-for-ci-run` label to the PR

Repeat all steps after any change to the PR.
- When the changes are ready to get merged — merge the original PR (not the internal one)

## Longer version:

GitHub Actions triggered by the `pull_request` event don't share repository secrets with the forks (for security reasons).
So, passing the CI pipeline on Pull Requests from external contributors is impossible.

We're using the following approach to make it work:
- After the review, assign the `approved-for-ci-run` label to the PR if changes look safe
- A GitHub Action will create an internal branch and a new PR with the same changes (for example, for a PR `#1234`, it'll be a branch `ci-run/pr-1234`)
- Because the PR is created from the internal branch, it is able to access repository secrets (that's why it's crucial to make sure that the PR doesn't contain any malicious code that could expose our secrets or intentionally harm the CI)
- The label gets removed automatically, so to run CI again with new changes, the label should be added again (after the review)

For details see [`approved-for-ci-run.yml`](.github/workflows/approved-for-ci-run.yml)
