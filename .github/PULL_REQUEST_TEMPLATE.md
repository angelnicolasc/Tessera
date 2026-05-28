## Summary

<!-- One paragraph: what does this PR do and why? -->

## Type

<!-- Pick one with [x]. -->

- [ ] feat — user-visible new functionality
- [ ] fix — bug fix
- [ ] refactor — internal change without behaviour difference
- [ ] perf — measurable performance improvement
- [ ] docs — documentation only
- [ ] test — adds or changes tests
- [ ] build / ci / chore

## Verification

<!-- Concretely how do you know this works? -->

- [ ] `cargo fmt && cargo clippy --workspace --all-targets -- -D warnings` clean
- [ ] `cargo test --workspace` passes locally
- [ ] `ruff check && ruff format --check && pyright` clean
- [ ] `pytest tests -v` passes locally
- [ ] New behaviour has a test or a stated reason it doesn't (e.g. perf-only change with bench)

## Related ADRs / issues

<!-- Link any ADR you authored or modified, plus issues this closes. -->

## Notes for reviewers

<!-- Areas of uncertainty, alternatives considered, deliberate scope cuts. -->
