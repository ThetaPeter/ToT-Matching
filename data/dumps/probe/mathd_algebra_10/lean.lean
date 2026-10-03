theorem mathd_algebra_10 : True := by
  have h1 : 1.2 * 30 = 36 := by
    norm_num
  have h2 : 1.3 * 20 = 26 := by
    norm_num
  have h3 : |(36 : ℝ) - 26| = 10 := by
    norm_num
  trivial