by
  have h : ∀ (a z : ℝ), 36 * a * z - 9 * a * a ≤ 36 * z * z := by
    intro a z
    have h1 : 0 ≤ 36 * (z - a / 2) ^ 2 := by
      nlinarith? [sq_nonneg (z - a / 2)]
    nlinarith
  exact ⟨⟩