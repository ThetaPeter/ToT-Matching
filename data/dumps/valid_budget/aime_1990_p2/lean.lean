theorem aime_1990_p2 : True := by
  have h1 : (3 + Real.sqrt 43) ^ 2 = 52 + 6 * Real.sqrt 43 := by
    ring_nf
    rw [show (Real.sqrt 43) ^ 2 = 43 from Real.pow_sqrt_eq_abs 43 |>.trans (abs_of_pos (by norm_num))]
    ring
  have h2 : (Real.sqrt 43 - 3) ^ 2 = 52 - 6 * Real.sqrt 43 := by
    ring_nf
    rw [show (Real.sqrt 43) ^ 2 = 43 from Real.pow_sqrt_eq_abs 43 |>.trans (abs_of_pos (by norm_num))]
    ring
  have h3 : Real.sqrt (52 + 6 * Real.sqrt 43) = 3 + Real.sqrt 43 := by
    rw [show 52 + 6 * Real.sqrt 43 = (3 + Real.sqrt 43) ^ 2 from by symm; exact h1]
    exact Real.sqrt_sq (show 0 ≤ 3 + Real.sqrt 43 from by positivity)
  have h4 : Real.sqrt (52 - 6 * Real.sqrt 43) = Real.sqrt 43 - 3 := by
    rw [show 52 - 6 * Real.sqrt 43 = (Real.sqrt 43 - 3) ^ 2 from by symm; exact h2]
    exact Real.sqrt_sq (sub_nonneg.mpr (by exact calc
      Real.sqrt 43 ≥ Real.sqrt 36 := Real.sqrt_le_sqrt (by norm_num)
      _ = 6 := by norm_num
      _ ≥ 3 := by norm_num))
  set α := 52 + 6 * Real.sqrt 43 with hα_def
  set β := 52 - 6 * Real.sqrt 43 with hβ_def
  have h5 : α^(3/2 : ℝ) = ((3 : ℝ) + Real.sqrt 43) ^ 3 := by
    rw [Real.rpow_div_two_eq_sqrt (show 0 ≤ α by
      intro h
      have := Real.sqrt_nonneg 43
      linarith)]
    rw [show Real.sqrt α = 3 + Real.sqrt 43 from h3]
    ring
  have h6 : β^(3/2 : ℝ) = (Real.sqrt 43 - 3) ^ 3 := by
    rw [Real.rpow_div_two_eq_sqrt (show 0 ≤ β by
      intro h
      have h43 : Real.sqrt 43 ≤ Real.sqrt 49 := Real.sqrt_le_sqrt (by norm_num)
      have : Real.sqrt 43 ≤ 7 := by linarith
      linarith)]
    rw [show Real.sqrt β = Real.sqrt 43 - 3 from h4]
    ring
  have h7 : ((3 : ℝ) + Real.sqrt 43) ^ 3 = 414 + 70 * Real.sqrt 43 := by
    ring_nf
    rw [show (Real.sqrt 43) ^ 2 = 43 from Real.pow_sqrt_eq_abs 43 |>.trans (abs_of_pos (by norm_num))]
    ring
  have h8 : (Real.sqrt 43 - 3) ^ 3 = -414 + 70 * Real.sqrt 43 := by
    ring_nf
    rw [show (Real.sqrt 43) ^ 2 = 43 from Real.pow_sqrt_eq_abs 43 |>.trans (abs_of_pos (by norm_num))]
    ring
  have h9 : α^(3/2 : ℝ) - β^(3/2 : ℝ) = 828 := by
    rw [h5, h6, h7, h8]
    ring
  trivial