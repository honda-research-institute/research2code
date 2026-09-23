---
name: code-role-classification
description: Use this when classifying paper elements or verifying code against paper elements. Defines the three code_role categories and their implications for code generation and verification.
---

## The Three Code Roles

Every technical element extracted from a paper MUST be assigned one of these three roles. The role determines what kind of code treatment the element needs and how verification judges the generated code.

### implement

This element defines a computation that should be directly coded. The code should match the formula or pseudocode as closely as possible.

**Examples:** Algorithm update rules, loss functions, gradient computations, training loops, data preprocessing steps.

**Typical types:** algorithm, equation (when the equation IS the method), experiment, hyperparameter.

### demonstrate

This element derives or explains something theoretical, but the underlying concept CAN and SHOULD be demonstrated empirically — through simulation, visualization, or numerical experiment. The code should illustrate the concept, not re-derive the math symbolically.

**Examples:**
- A derivation showing why bias correction is needed → demonstrate by simulating the bias
- A regret definition used to state a convergence result → demonstrate by computing regret empirically and plotting its growth rate
- An equation showing the expected value of an estimator → demonstrate by running Monte Carlo simulations

**Typical types:** equation (when part of a derivation), concept, property (when empirically validatable).

### theoretical

This element is purely theoretical — a proof step, a lemma used inside a derivation, an intermediate inequality, or a bound that serves as scaffolding for a larger proof. There is no meaningful way to implement or demonstrate it in code.

**Examples:** Proof lemmas, intermediate inequalities in convergence proofs, definitions used only within proofs, theoretical upper/lower bounds that cannot be checked empirically.

**Typical types:** property (proof-internal), equation (proof-internal).

## Choosing Between demonstrate and theoretical

Ask: "Could a practitioner write a simulation or experiment that shows this concept in action?"

- If **yes** → `demonstrate`
- If the element only makes sense as a step in a pen-and-paper proof → `theoretical`

## Verification Implications by Role

### implement equations
- Find the code that implements this formula
- Check: Does the code correctly implement it? Are all terms present? Are operations correct?
- **PASS** if code faithfully implements the formula
- **WARNING** if code implements it with minor modifications (e.g., epsilon for numerical stability)
- **FAIL** only if code has a clear error, or if no code implements it at all

### demonstrate equations
- These are theoretical derivations or expectations. The code should DEMONSTRATE the concept through simulation/visualization, not re-derive it symbolically.
- Look for: Monte Carlo simulations, empirical comparisons, plots showing predicted behavior, numerical experiments validating the claim.
- **PASS** if the code demonstrates the concept effectively, even if it never literally computes the exact formula terms
- **WARNING** if the demonstration is partial or could be more convincing
- **FAIL** only if the code claims to demonstrate the concept but does so incorrectly, or if there is no code related to this concept at all

### theoretical equations
- Code is not expected for these.
- **PASS** with message "Theoretical — no code implementation expected"
- **FAIL** only if code attempts to implement it and does so incorrectly
