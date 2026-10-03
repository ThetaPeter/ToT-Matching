by
  have h : ∀ a : ℝ, 10 * a ≤ 28 * a ^ 2 + 1 := by
    intro a
    have h1 : 0 ≤ 28 * (a - 5/28) ^ 2 := by
      nlinarith? [sq_nonneg (a - 5/28)]
    nlinarith
  trivial