theorem amc12b_2004_p3 : True := by
  have h1296 : 1296 = 6^4 := by norm_num
  have h6 : 6 = 2 * 3 := by norm_num
  have h_factorization : 1296 = 2^4 * 3^4 := by
    calc
      1296 = 6^4 := by norm_num
      _ = (2 * 3)^4 := by rw [h6]
      _ = 2^4 * 3^4 := by ring
  have h_unique : ∀ (x y : ℕ), 2^x * 3^y = 1296 → x = 4 ∧ y = 4 := by
    intro x y h
    have h_pos : x > 0 ∧ y > 0 := by
      constructor
      · by_contra! H
        have : 2^x * 3^y ≤ 1 * 3^y := by
          refine mul_le_mul_right' (pow_le_pow_right (by norm_num) H) _
        have : 3^y ≤ 1296 := by linarith
        have : y ≤ 6 := by
          have : 3^6 = 729 := by norm_num
          have : 3^7 = 2187 := by norm_num
          omega
        have : 2^x * 3^y ≤ 1 * 729 := by
          refine mul_le_mul ?_ (pow_le_pow_right (by norm_num) this) (by norm_num) (by norm_num)
          exact pow_le_pow_right (by norm_num) H
        norm_num at this h
        linarith
      · by_contra! H
        have : 2^x * 3^y ≤ 2^x * 1 := by
          refine mul_le_mul_left' (pow_le_pow_right (by norm_num) H) _
        have : 2^x ≤ 1296 := by linarith
        have : x ≤ 10 := by
          have : 2^10 = 1024 := by norm_num
          have : 2^11 = 2048 := by norm_num
          omega
        have : 2^x * 3^y ≤ 1024 * 1 := by
          refine mul_le_mul (pow_le_pow_right (by norm_num) this) (pow_le_pow_right (by norm_num) H) 
            (by norm_num) (by norm_num)
        norm_num at this h
        linarith
    have hx : x ≤ 4 := by
      by_contra! H
      have : 2^x ≥ 2^5 := pow_le_pow_right (by norm_num) H
      have : 2^5 = 32 := by norm_num
      have : 3^y ≥ 3^1 := pow_le_pow_right (by norm_num) (Nat.one_le_iff_ne_zero.mpr h_pos.right.ne')
      have : 3^1 = 3 := by norm_num
      have : 2^x * 3^y ≥ 32 * 3 := by exact mul_le_mul this this (by norm_num) (by norm_num)
      norm_num at this h
      linarith
    have hy : y ≤ 4 := by
      by_contra! H
      have : 3^y ≥ 3^5 := pow_le_pow_right (by norm_num) H
      have : 3^5 = 243 := by norm_num
      have : 2^x ≥ 2^1 := pow_le_pow_right (by norm_num) (Nat.one_le_iff_ne_zero.mpr h_pos.left.ne')
      have : 2^1 = 2 := by norm_num
      have : 2^x * 3^y ≥ 2 * 243 := by exact mul_le_mul this this (by norm_num) (by norm_num)
      norm_num at this h
      linarith
    have hx' : x ≥ 4 := by
      by_contra! H
      have : 2^x ≤ 2^3 := pow_le_pow_right (by norm_num) (Nat.lt_succ.mp H)
      have : 2^3 = 8 := by norm_num
      have : 3^y ≤ 3^4 := pow_le_pow_right (by norm_num) hy
      have : 3^4 = 81 := by norm_num
      have : 2^x * 3^y ≤ 8 * 81 := by exact mul_le_mul this this (by norm_num) (by norm_num)
      norm_num at this h
      linarith
    have hy' : y ≥ 4 := by
      by_contra! H
      have : 3^y ≤ 3^3 := pow_le_pow_right (by norm_num) (Nat.lt_succ.mp H)
      have : 3^3 = 27 := by norm_num
      have : 2^x ≤ 2^4 := pow_le_pow_right (by norm_num) hx
      have : 2^4 = 16 := by norm_num
      have : 2^x * 3^y ≤ 16 * 27 := by exact mul_le_mul this this (by norm_num) (by norm_num)
      norm_num at this h
      linarith
    exact ⟨Nat.le_antisymm hx hx', Nat.le_antisymm hy hy'⟩
  have h_sum : ∀ (x y : ℕ), 2^x * 3^y = 1296 → x + y = 8 := by
    intro x y h
    have hxy := h_unique x y h
    rcases hxy with ⟨rfl, rfl⟩
    norm_num
  trivial