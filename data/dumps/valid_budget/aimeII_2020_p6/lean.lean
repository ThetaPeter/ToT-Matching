theorem aimeII_2020_p6 : True := by
  have tc_1 : ∀ (n : ℕ) (h : n ≥ 3), t (n : ℕ) = (5 * t (n - 1) + 1) / (25 * t (n - 2)) := by
    intro n hn
    sorry
  have l1 : ∀ (n : ℕ) (h : n ≥ 3), 25 * t n * t (n - 2) - 5 * t (n - 1) = 1 := by
    intro n hn
    have := tc_1 n hn
    sorry
  have l2 : ∀ (n : ℕ) (h : n ≥ 1), t (n + 6) = t n := by
    intro n hn
    have := l1 (n + 3) (by omega)
    have := l1 (n + 4) (by omega)
    have := l1 (n + 5) (by omega)
    sorry
  have ts_1 : t 2020 = p / q ∧ Nat.gcd p q = 1 → p + q = 626 := by
    intro h
    rcases h with ⟨ht, hgcd⟩
    have period : t 2020 = t (2020 % 6) := by
      have := l2 (2020 % 6) (by omega)
      sorry
    sorry
  trivial