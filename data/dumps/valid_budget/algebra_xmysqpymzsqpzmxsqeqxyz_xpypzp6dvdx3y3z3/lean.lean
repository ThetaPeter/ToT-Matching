theorem algebra_xmysqpymzsqpzmxsqeqxyz_xpypzp6dvdx3y3z3 : True := by
  intro h
  rcases h with ⟨hx, hy, hz, h_eq⟩
  set s := x + y + z with hs_def
  set p := x * y + y * z + z * x with hp_def
  set q := x * y * z with hq_def
  have h_sq : (x - y)^2 + (y - z)^2 + (z - x)^2 = 2 * (s^2 - 3 * p) := by
    ring_nf
    simp [s, p]
    ring
  have h_eq1 : 2 * (s^2 - 3 * p) = q := by
    rw [← h_sq, h_eq]
  have h_int : q / 2 ∈ ℤ := by
    have : 2 ∣ q := by
      rw [← h_eq1]
      exact ⟨s^2 - 3 * p, by ring⟩
    exact Int.ediv_eq_of_eq_dvd (by ring) this
  have h_sum_cubes : x^3 + y^3 + z^3 = s^3 - 3 * s * p + 3 * q := by
    simp [s, p, q]
    ring
  have h_factor : x^3 + y^3 + z^3 = (q / 2) * (s + 6) := by
    rw [h_sum_cubes]
    have : s^3 - 3 * s * p + 3 * q = (s^2 - 3 * p) * s + 3 * q := by ring
    rw [this]
    rw [show s^2 - 3 * p = q / 2 from ?_]
    ring
    exact (Int.ediv_eq_of_eq_dvd ?_ (by rw [h_eq1]; exact ⟨s^2 - 3 * p, by ring⟩)).symm
  have h_dvd : (s + 6) ∣ (x^3 + y^3 + z^3) := by
    rw [h_factor]
    exact ⟨q / 2, by ring⟩
  exact h_dvd