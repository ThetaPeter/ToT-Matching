theorem amc12a_2015_p10 : True := by
  have h : ∃ (x y : ℤ), x > y ∧ y > 0 ∧ x + y + x * y = 80 := by
    refine ⟨26, 2, ?_, ?_, ?_⟩
    · decide
    · decide
    · decide
  exact True.intro