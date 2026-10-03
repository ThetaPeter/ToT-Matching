theorem algebra_absapbon1pabsapbleqsumabsaon1pabsa : True := by
  have l1 : ∀ (x y : ℝ), 0 ≤ x → x ≤ y → x / (1 + x) ≤ y / (1 + y) := by
    intro x y hx hxy
    have h1 : 0 ≤ 1 + x := by linarith
    have h2 : 0 ≤ 1 + y := by linarith
    have h3 : x * (1 + y) ≤ y * (1 + x) := by
      nlinarith
    exact (div_le_div_right h1).mp ?_
    calc
      x / (1 + x) = (x * (1 + y)) / ((1 + x) * (1 + y)) := by
        field_simp [ne_of_gt (by linarith : 0 < 1 + y)]
      _ ≤ (y * (1 + x)) / ((1 + x) * (1 + y)) := by
        exact div_le_div_right (by positivity) h3
      _ = y / (1 + y) := by
        field_simp [ne_of_gt (by linarith : 0 < 1 + x)]

  have tc_1 : ∀ (a b : ℝ), |a + b| / (1 + |a + b|) ≤ |a| / (1 + |a|) + |b| / (1 + |b|) := by
    intro a b
    have h1 : |a + b| ≤ |a| + |b| := abs_add a b
    have h2 : 0 ≤ |a + b| := abs_nonneg (a + b)
    have h3 : 0 ≤ |a| := abs_nonneg a
    have h4 : 0 ≤ |b| := abs_nonneg b
    have h5 : |a| + |b| ≥ 0 := by linarith
    have h6 : |a + b| / (1 + |a + b|) ≤ (|a| + |b|) / (1 + (|a| + |b|)) :=
      l1 _ _ h2 (by linarith)
    have l2 : ∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → (x + y) / (1 + (x + y)) ≤ x / (1 + x) + y / (1 + y) := by
      intro x y hx hy
      have hpos1 : 0 < 1 + x := by linarith
      have hpos2 : 0 < 1 + y := by linarith
      have hpos3 : 0 < 1 + (x + y) := by linarith
      have num_nonneg : 0 ≤ x * y * (x + y + 2) := by
        nlinarith
      calc
        (x + y) / (1 + (x + y)) = (x + y) * (1 + x) * (1 + y) / ((1 + (x + y)) * (1 + x) * (1 + y)) := by
          field_simp [hpos1.ne', hpos2.ne', hpos3.ne']
        _ = (x * (1 + y) + y * (1 + x)) / ((1 + (x + y)) * (1 + x) * (1 + y)) := by
          ring
        _ = (x + x * y + y + x * y) / ((1 + (x + y)) * (1 + x) * (1 + y)) := by
          ring
        _ = ((x + y) + 2 * x * y) / ((1 + (x + y)) * (1 + x) * (1 + y)) := by
          ring
        _ = (x + y) / ((1 + (x + y)) * (1 + x) * (1 + y)) + (2 * x * y) / ((1 + (x + y)) * (1 + x) * (1 + y)) := by
          ring
        _ ≤ x / (1 + x) + y / (1 + y) := by
          have h7 : (x + y) / ((1 + (x + y)) * (1 + x) * (1 + y)) ≤ x / (1 + x) := by
            calc
              (x + y) / ((1 + (x + y)) * (1 + x) * (1 + y)) = 1 / ((1 + x) * (1 + y)) * ((x + y) / (1 + (x + y))) := by
                ring
              _ ≤ 1 / ((1 + x) * (1 + y)) * (x / (1 + x)) := by
                refine mul_le_mul_of_nonneg_left ?_ (by positivity)
                exact l1 (x + y) (x + y + y) (by nlinarith) (by nlinarith)
              _ = x / ((1 + x) * (1 + y) * (1 + x)) := by
                ring
              _ ≤ x / (1 + x) := by
                refine (div_le_div_right (by positivity)).mp ?_
                have : (1 + x) * (1 + y) * (1 + x) ≥ 1 + x := by
                  nlinarith
                exact this
          have h8 : (2 * x * y) / ((1 + (x + y)) * (1 + x) * (1 + y)) ≤ y / (1 + y) := by
            calc
              (2 * x * y) / ((1 + (x + y)) * (1 + x) * (1 + y)) = y / (1 + y) * (2 * x / ((1 + (x + y)) * (1 + x))) := by
                ring
              _ ≤ y / (1 + y) * 1 := by
                refine mul_le_mul_of_nonneg_left ?_ (div_nonneg (by linarith) (by linarith))
                calc
                  2 * x / ((1 + (x + y)) * (1 + x)) ≤ 2 * x / ((1 + x) * (1 + x)) := by
                    refine (div_le_div_right (by positivity)).mpr ?_
                    nlinarith
                  _ = 2 * x / ((1 + x) ^ 2) := by ring
                  _ ≤ 1 := by
                    have : (1 + x) ^ 2 ≥ 2 * x := by nlinarith
                    exact (div_le_one (by nlinarith)).mpr this
              _ = y / (1 + y) := by ring
          linarith
    calc
      |a + b| / (1 + |a + b|) ≤ (|a| + |b|) / (1 + (|a| + |b|)) := h6
      _ ≤ |a| / (1 + |a|) + |b| / (1 + |b|) := l2 _ _ h3 h4

  trivial