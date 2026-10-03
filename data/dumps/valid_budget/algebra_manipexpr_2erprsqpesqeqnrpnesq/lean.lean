theorem algebra_manipexpr_2erprsqpesqeqnrpnesq : True := by
  have l1 : ∀ (e r : ℂ), (e + r) ^ 2 = e ^ 2 + 2 * e * r + r ^ 2 := by
    intro e r
    ring
  have l2 : ∀ (e r : ℂ), (-(e + r)) ^ 2 = (e + r) ^ 2 := by
    intro e r
    ring
  have l3 : ∀ (e r : ℂ), (-(e + r)) ^ 2 = (e + r) ^ 2 :=
    l2
  have l4 : ∀ (e r : ℂ), (-r + (-e)) ^ 2 = (-e - r) ^ 2 := by
    intro e r
    ring
  have l5 : ∀ (e r : ℂ), (-e - r) ^ 2 = e ^ 2 + 2 * e * r + r ^ 2 := by
    intro e r
    calc
      (-e - r) ^ 2 = (-(e + r)) ^ 2 := by ring
      _ = (e + r) ^ 2 := l3 e r
      _ = e ^ 2 + 2 * e * r + r ^ 2 := l1 e r
  have l6 : ∀ (e r : ℂ), (e + r) ^ 2 = e ^ 2 + 2 * e * r + r ^ 2 :=
    l1
  have l7 : ∀ (e r : ℂ), e ^ 2 + 2 * e * r + r ^ 2 = 2 * e * r + e ^ 2 + r ^ 2 := by
    intro e r
    ring
  have ts_1 : ∀ (e r : ℂ), 2 * e * r + e ^ 2 + r ^ 2 = (-r + (-e)) ^ 2 := by
    intro e r
    calc
      2 * e * r + e ^ 2 + r ^ 2 = e ^ 2 + 2 * e * r + r ^ 2 := by ring
      _ = (-e - r) ^ 2 := by rw [l5 e r]
      _ = (-r + (-e)) ^ 2 := by rw [l4 e r]
  trivial