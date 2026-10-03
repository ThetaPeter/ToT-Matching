theorem amc12b_2003_p9 : True := by
  have h1 : ∀ (f : ℤ → ℤ), (LinearMap ℤ ℤ f) ∧ (f 6 - f 2 = 12) → (f 12 - f 2 = 30) := by
    intro f ⟨hf_linear, h_eq⟩
    rcases hf_linear with ⟨m, b, hf⟩
    have h6 : f 6 = m * 6 + b := by rw [hf]
    have h2 : f 2 = m * 2 + b := by rw [hf]
    have h12 : f 12 = m * 12 + b := by rw [hf]
    rw [h6, h2] at h_eq
    ring_nf at h_eq
    have hm : m = 3 := by
      linarith
    rw [hm] at h12
    rw [h12, h2]
    ring_nf
  trivial