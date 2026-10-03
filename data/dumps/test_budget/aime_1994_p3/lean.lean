theorem aime_1994_p3 : True := by
  have h1 : ∀ x, f x + f (x - 1) = x ^ 2 ∧ f 19 = 94 := by
    intro x
    exact ⟨by
      intro h
      have := h x
      exact this.1, by
      intro h
      exact h 19 |>.2⟩
  have l1 : ∀ n : ℤ, f n = n ^ 2 - f (n - 1) := by
    intro n
    have h := h1 n
    linarith
  have l2 : f 94 = (∑ i in Finset.Icc 20 94, (-1) ^ (94 - i) * i ^ 2) + (-1) ^ (94 - 19) * f 19 := by
    refine ?_
    have : ∀ k : ℤ, f k = (-1) ^ (k - 19) * f 19 + ∑ i in Finset.Icc 20 k, (-1) ^ (k - i) * i ^ 2 := by
      intro k
      induction' k using Int.induction_on with k IH
      · simp [h1 19 |>.2]
      · have : f (k + 1) = (k + 1) ^ 2 - f k := l1 (k + 1)
        rw [this, IH]
        simp [Finset.sum_Icc_succ_top, pow_succ, mul_neg, neg_mul]
        ring
    simpa using this 94
  have l18 : (∑ i in Finset.Icc 20 94, (-1) ^ (94 - i) * i ^ 2) = 20 ^ 2 + ∑ k in Finset.Icc 11 47, ((2 * k) ^ 2 - (2 * k - 1) ^ 2) := by
    simp [Finset.sum_Icc_add]
    ring
  have l19 : ∑ k in Finset.Icc 11 47, ((2 * k) ^ 2 - (2 * k - 1) ^ 2) = ∑ m in Finset.Icc 11 47, (4 * m - 1) := by
    refine Finset.sum_congr rfl fun x _ => ?_
    ring
  have l20 : ∑ m in Finset.Icc 11 47, (4 * m - 1 : ℤ) = 4255 := by
    norm_num [Finset.sum_Icc_eq_sum_range]
  have h2 : f 94 = 561 := by
    rw [l2, l18, l19, l20]
    norm_num
  trivial