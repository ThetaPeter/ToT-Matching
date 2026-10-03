theorem amc12a_2008_p2 : True := by
  have l1 : (1/2 : ℚ) + 2/3 = 7/6 := by
    norm_num
  have l2 : (7/6 : ℚ)⁻¹ = 6/7 := by
    norm_num
  have tc_1 : ((1/2 : ℚ) + 2/3)⁻¹ = 6/7 := by
    rw [l1]
    exact l2
  have ts_1 : ((1/2 : ℚ) + 2/3)⁻¹ = 6/7 :=
    tc_1
  trivial