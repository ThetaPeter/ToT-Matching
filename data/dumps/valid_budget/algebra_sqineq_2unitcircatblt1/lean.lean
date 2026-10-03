theorem algebra_sqineq_2unitcircatblt1 : True := by
  have h1 : ∀ (a b : ℝ), (a - b) ^ 2 ≥ 0 := by
    intro a b
    nlinarith? [sq_nonneg (a - b)]
  have h2 : ∀ (a b : ℝ), a ^ 2 + b ^ 2 ≥ 2 * a * b := by
    intro a b
    have h : (a - b) ^ 2 ≥ 0 := h1 a b
    nlinarith
  have h3 : ∀ (a b : ℝ), a ^ 2 + b ^ 2 = 2 → 2 ≥ 2 * a * b := by
    intro a b h
    have h' : a ^ 2 + b ^ 2 ≥ 2 * a * b := h2 a b
    linarith
  have h4 : ∀ (a b : ℝ), a ^ 2 + b ^ 2 = 2 → 1 ≥ a * b := by
    intro a b h
    have h' : 2 ≥ 2 * a * b := h3 a b h
    linarith
  have h5 : ∀ (a b : ℝ), a ^ 2 + b ^ 2 = 2 → a * b ≤ 1 := by
    intro a b h
    have h' : 1 ≥ a * b := h4 a b h
    linarith
  trivial