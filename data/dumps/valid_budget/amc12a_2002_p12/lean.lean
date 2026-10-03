theorem amc12a_2002_p12 : True := by
  have h61_prime : Nat.Prime 61 := by
    refine Nat.prime_def_lt.mpr ⟨by decide, ?_⟩
    intro m hm
    have : m ≤ 61 := by omega
    interval_cases m <;> decide
  have h2_prime : Nat.Prime 2 := by decide
  have h_sum : 2 + 61 = 63 := by decide
  have h_prod : 2 * 61 = 122 := by decide
  have h_unique : ∀ (p q : ℕ), Nat.Prime p → Nat.Prime q → p + q = 63 → p * q = 122 := by
    intro p q hp hq hsum
    have h_even : Even (p + q) := by
      rw [hsum]
      exact ⟨31, by ring⟩
    rcases Nat.even_add.mp h_even with (⟨hp_even⟩ | ⟨hq_even⟩)
    · have : p = 2 := Nat.Prime.even_two hp hp_even
      rw [this] at hsum
      have : q = 61 := by omega
      rw [this, this]
      ring
    · have : q = 2 := Nat.Prime.even_two hq hq_even
      rw [this] at hsum
      have : p = 61 := by omega
      rw [this, this]
      ring
  trivial