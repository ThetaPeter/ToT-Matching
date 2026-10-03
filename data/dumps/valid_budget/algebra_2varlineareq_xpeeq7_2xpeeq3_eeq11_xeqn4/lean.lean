theorem algebra_2varlineareq_xpeeq7_2xpeeq3_eeq11_xeqn4 : True := by
  have h1 : ∀ (x e : ℤ), x + e = 7 ∧ 2 * x + e = 3 → e = 11 ∧ x = -4 := by
    intro x e ⟨h2, h3⟩
    have h4 : x = -4 := by
      linarith
    have h5 : e = 11 := by
      linarith
    exact ⟨h5, h4⟩
  trivial