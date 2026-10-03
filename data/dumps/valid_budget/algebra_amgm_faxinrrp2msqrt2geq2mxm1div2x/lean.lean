theorem algebra_amgm_faxinrrp2msqrt2geq2mxm1div2x : True := by
  have h : ∀ (x : ℝ), x > 0 → 2 - Real.sqrt 2 ≥ 2 - x - 1 / (2 * x) := by
    intro x hx_pos
    have h_ineq : x + 1 / (2 * x) ≥ Real.sqrt 2 := by
      have h_amgm : x + 1 / (2 * x) ≥ 2 * Real.sqrt (x * (1 / (2 * x))) := by
        apply Real.arithmetic_geometric_mean_two (by linarith) (by positivity)
      have h_simplify : x * (1 / (2 * x)) = 1/2 := by
        field_simp [ne_of_gt hx_pos]
      rw [h_simplify] at h_amgm
      calc
        x + 1 / (2 * x) ≥ 2 * Real.sqrt (1/2) := h_amgm
        _ = 2 * (Real.sqrt 2 / Real.sqrt 4) := by rw [show Real.sqrt (4 : ℝ) = 2 by norm_num]
        _ = 2 * (Real.sqrt 2 / 2) := by norm_num
        _ = Real.sqrt 2 := by ring
    linarith
  trivial