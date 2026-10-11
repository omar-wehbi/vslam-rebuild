# How the work was run

I built this project with AI agents (Claude models, run through Claude Code) inside a verification loop I designed.
The agents planned, reviewed and wrote the code. Whether a stage passed was decided by scripts, not by an agent. I
made the decisions of substance. This file describes the loop; the closing report, section 9, gives its record.

The loop's code (the controller, the verifier and their tests) is kept in the private working repository.

---

## The roles

| Role | Agent or script | What it did | What it could change |
|---|---|---|---|
| **Orchestrator** | AI agent | Wrote each stage contract: the spec, the executable checks, and the pass/fail thresholds, all fixed before any code. After the stage, read the combined log and proposed pass, retry or replan | Stage contracts only |
| **Theorist** | AI agent | Before the build: does the contract match the method notes? After it: does the diff match the contract and the notes? | Nothing (read-only) |
| **Executor** | AI agent | Built the stage and reported what it did | Its own git worktree, minus protected paths |
| **Verifier** | script | Ran every check of the contract against the executor's exact commit and computed pass or fail | Its own result files |
| **Controller** | script | Ran the same cycle for every stage, applied the gate rules, merged and tagged verified work | Git and the run records |

Outside the loop sat the **theory workshop**: an interactive AI session in which I read the papers, argued the
method and wrote the notes the theorist judges against. Each decision taken there was recorded as a numbered ruling
and took effect only once I ratified it. Keeping the workshop out of the loop meant the theorist never graded a
stage against notes it had written in the same session.

Each agent call was a fresh session with a restricted tool allowlist and a JSON schema for its answer. An answer
that failed its schema was asked for once more, then sent to me.

---

## One stage, step by step

1. **Theorist, before the build.** If it rejects the contract, the orchestrator gets one revision; a second
   rejection stops the loop for me.
2. **Lock.** The approved contract is written once and hashed. Every later step reads that file, and a changed hash
   stops the loop. The method notes are frozen with it, so the post-build review grades against the same text.
3. **Build.** The executor works in an isolated git worktree. A hook blocks edits to protected paths (the harness,
   the notes, the plans, the run records), and a diff check afterwards is the backstop.
4. **Verify.** The verifier runs from the main checkout, not the stage branch, so the executor cannot edit its own
   exam. Each check prints `METRIC name=value` lines, and pass or fail is computed from them against the contract's
   thresholds.
5. **Theorist, after the build.** It sees the locked contract, the diff and the file list, and must cite headings
   that exist in the frozen notes and cover every changed file.
6. **Orchestrator.** It reads one combined log of everything above and proposes pass, retry or replan.
7. **Gates.** The controller applies its rules. If none fire, it merges the stage and tags it verified. The set of
   those tags is the definition of "verified".

---

## When the loop stopped for me

Only on disagreement, or when something would weaken the plan or touch accepted work. Among the rules:
- the orchestrator says pass but the verifier failed, or the theorist found a deviation;
- the orchestrator removed a check, loosened a threshold, removed a dependency or edited a locked contract (the
  edit is reverted);
- a contract names a check file that does not exist yet (harness work comes to me first);
- a retry beyond the stage's budget;
- the main branch changed code while the stage was being built.

My ruling went into a decision file (accept, retry, fail the stage, allow or refuse a plan change, and so on), and
my note was appended to the log the orchestrator reads next.

---

## What was enforced, and what was only judged

The loop was honest about this split:
- **Enforced by code, and tested:** the fixed cycle, the locked contract, the protected paths, pass or fail, the
  plan-weakening rules, the gates and the evidence kept per attempt. A suite of 42 tests drives the controller with
  deliberately misbehaving stub roles: an executor that edits the harness, an orchestrator that loosens a
  threshold, a theorist that cites nothing. Each one must be caught.
- **Judged by a model, checked only for form:** whether a contract or a diff matches the method, whether a stage
  should pass, retry or replan, and whether the executor's own report is honest.
- **Not enforced mechanically:** that later contracts carry the earlier stages' regression checks, stage size, and
  the quality of a check itself. Those rested on the orchestrator's instructions and on me.

---

## What the record showed

45 stage contracts between 2026-09-17 and 2026-10-11: 33 verified, 12 failed and replanned.
- **None of the failures was a coding error.** In every case the executor built what the contract asked. The
  causes were the harness, a wrong bar or premise in the contract, a real measurement no bar covered, or a design
  withdrawn before the build.
- **Two verified stages carried a bar that failed by construction.** Both were caught only in the harness dry run,
  where a contract's checks are run on a scratch checkout, not by the loop itself.

The weak link was writing the specification, not building it. Because a script decided pass or fail against a
contract locked before the build, a wrong specification showed up as a failed stage with a recorded cause, not as
code quietly bent to fit.
