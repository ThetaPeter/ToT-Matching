theorem aime_1997_p11 : True := by
  have hsqrt2_lower : (1.41421 : ℝ) ^ 2 < 2 := by norm_num
  have hsqrt2_upper : (1.41422 : ℝ) ^ 2 > 2 := by norm_num
  have hsqrt2_pos : 0 < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num : (0 : ℝ) < 2)
  have hsqrt2_gt : 1.41421 < Real.sqrt 2 := by
    exact Real.sqrt_lt_sqrt (by norm_num) hsqrt2_lower
  have hsqrt2_lt : Real.sqrt 2 < 1.41422 := by
    exact Real.sqrt_lt_sqrt (by norm_num) hsqrt2_upper
  have hx_eq : Real.sqrt 2 + 1 = Real.sqrt 2 + 1 := rfl
  have h100x_eq : 100 * (Real.sqrt 2 + 1) = 100 * (Real.sqrt 2 + 1) := rfl
  have h100x_gt : 241.421 < 100 * (Real.sqrt 2 + 1) := by
    linarith [hsqrt2_gt]
  have h100x_lt : 100 * (Real.sqrt 2 + 1) < 241.422 := by
    linarith [hsqrt2_lt]
  have hfloor_eq : ⌊100 * (Real.sqrt 2 + 1)⌋ = 241 := by
    have : (241 : ℝ) ≤ 100 * (Real.sqrt 2 + 1) := by linarith [hsqrt2_gt]
    have : 100 * (Real.sqrt 2 + 1) < (242 : ℝ) := by linarith [hsqrt2_lt]
    exact Int.floor_eq_iff.mpr ⟨by linarith, by linarith⟩
  trivial