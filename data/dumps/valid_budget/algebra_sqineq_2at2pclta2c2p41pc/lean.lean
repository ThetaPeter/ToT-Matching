theorem algebra_sqineq_2at2pclta2c2p41pc : True := by
  have h : ∀ (a c : ℝ), 2 * a * (2 + c) ≤ a ^ 2 + c ^ 2 + 4 * (1 + c) := by
    intro a c
    have h1 : a ^ 2 + c ^ 2 + 4 * (1 + c) - 2 * a * (2 + c) = (a - 2 - c) ^ 2 := by
      ring_nf
    have h2 : 0 ≤ (a - 2 - c) ^ 2 := by
      nlinarith [sq_nonneg (a - 2 - c)]
    linarith
  trivial