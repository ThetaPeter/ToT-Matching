theorem aime_1987_p5 : True := by
  have h1 : 507 = 3 * 13 * 13 := by norm_num
  have divisors : {d : ℤ | d > 0 ∧ d ∣ 507} = ({1, 3, 13, 39, 169, 507} : Set ℤ) := by
    ext d
    constructor
    · intro ⟨hd_pos, hd⟩
      have := show d ∣ 507 from hd
      have : d ∣ 3 * 13 * 13 := by rw [h1] at this; exact this
      have prime_factors : {p : ℤ | p.Prime ∧ p ∣ d} ⊆ ({3, 13} : Set ℤ) := by
        intro p ⟨hp, hp_dvd⟩
        have : p ∣ 3 * 13 * 13 := Nat.dvd_trans hp_dvd this
        rcases hp.dvd_mul.mp this with (h | h)
        · have : p ∣ 3 := h
          have : p = 3 := by
            apply hp.eq_one_or_self_of_dvd 3
            exact h
            norm_num
          simp [this]
        · rcases hp.dvd_mul.mp h with (h' | h')
          · have : p ∣ 13 := h'
            have : p = 13 := by
              apply hp.eq_one_or_self_of_dvd 13
              exact h'
              norm_num
            simp [this]
          · have : p ∣ 13 := h'
            have : p = 13 := by
              apply hp.eq_one_or_self_of_dvd 13
              exact h'
              norm_num
            simp [this]
      have : d = 1 ∨ d = 3 ∨ d = 13 ∨ d = 39 ∨ d = 169 ∨ d = 507 := by
        have : d.natAbs ∣ 507 := by
          rw [← Int.coe_nat_dvd]
          exact hd
        have : d.natAbs ∈ ({1, 3, 13, 39, 169, 507} : Set ℕ) := by
          have : 507 = 3 * 13 * 13 := by norm_num
          simp_rw [Finset.mem_insert, Finset.mem_singleton]
          exact Nat.mem_divisors.mp ⟨this ▸ hd, by norm_num⟩
        simp at this
        rcases this with (rfl|rfl|rfl|rfl|rfl|rfl)
        · left; rfl
        · right; left; rfl
        · right; right; left; rfl
        · right; right; right; left; rfl
        · right; right; right; right; left; rfl
        · right; right; right; right; right; rfl
      simp [this, hd_pos]
    · intro h
      simp at h
      rcases h with (rfl|rfl|rfl|rfl|rfl|rfl)
      · exact ⟨by norm_num, by norm_num⟩
      · exact ⟨by norm_num, by norm_num⟩
      · exact ⟨by norm_num, by norm_num⟩
      · exact ⟨by norm_num, by norm_num⟩
      · exact ⟨by norm_num, by norm_num⟩
      · exact ⟨by norm_num, by norm_num⟩
  have l6 : {d : ℤ | d > 0 ∧ d ∣ 507} = ({1, 3, 13, 39, 169, 507} : Set ℤ) := divisors
  have l7 : {d : ℤ | d > 0 ∧ d ∣ 507 ∧ d ≡ 1 [ZMOD 3]} = ({1, 13, 169} : Set ℤ) := by
    ext d
    constructor
    · intro ⟨hd_pos, hd, hd_mod⟩
      have : d ∈ ({1, 3, 13, 39, 169, 507} : Set ℤ) := by
        rw [← l6]
        exact ⟨hd_pos, hd⟩
      simp at this
      rcases this with (rfl|rfl|rfl|rfl|rfl|rfl)
      · simp [show (1 : ℤ) ≡ 1 [ZMOD 3] from by decide]
      · exfalso
        have : (3 : ℤ) ≡ 1 [ZMOD 3] := hd_mod
        norm_num [Int.modEq_iff_modEq_natAbs] at this
      · simp [show (13 : ℤ) ≡ 1 [ZMOD 3] from by decide]
      · exfalso
        have : (39 : ℤ) ≡ 1 [ZMOD 3] := hd_mod
        norm_num [Int.modEq_iff_modEq_natAbs] at this
      · simp [show (169 : ℤ) ≡ 1 [ZMOD 3] from by decide]
      · exfalso
        have : (507 : ℤ) ≡ 1 [ZMOD 3] := hd_mod
        norm_num [Int.modEq_iff_modEq_natAbs] at this
    · intro h
      simp at h
      rcases h with (rfl|rfl|rfl)
      · exact ⟨by norm_num, by norm_num, by decide⟩
      · exact ⟨by norm_num, by norm_num, by decide⟩
      · exact ⟨by norm_num, by norm_num, by decide⟩
  have tc_1 : ∃ (x y : ℤ), y^2 + 3*x^2*y^2 = 30*x^2 + 517 := by
    refine ⟨2, 7⟩
    norm_num
  have l1 : ∀ (x y : ℤ), y^2 + 3*x^2*y^2 = 30*x^2 + 517 → (3*x^2 + 1)*(y^2 - 10) = 507 := by
    intro x y h
    have : y^2 + 3*x^2*y^2 - 30*x^2 - 517 = 0 := by linarith
    have : (3*x^2 + 1)*(y^2 - 10) = 507 := by
      nlinarith
    exact this
  have l3 : ∀ (x y : ℤ), (3*x^2 + 1)*(y^2 - 10) = 507 → 3*x^2 + 1 ∣ 507 ∧ 3*x^2 + 1 ≥ 1 := by
    intro x y h
    constructor
    · refine ⟨y^2 - 10, ?_⟩
      linarith
    · nlinarith [sq_nonneg (x : ℤ)]
  have l8 : ∀ (x : ℤ), (3*x^2 + 1 ∣ 507 ∧ 3*x^2 + 1 ≥ 1) → 3*x^2 + 1 ∈ ({1, 13, 169} : Set ℤ) := by
    intro x ⟨h_dvd, h_ge⟩
    have : 3*x^2 + 1 > 0 := by linarith
    have : 3*x^2 + 1 ≡ 1 [ZMOD 3] := by
      have : 3*x^2 ≡ 0 [ZMOD 3] := by
        apply Int.modEq_zero_iff_dvd.mpr
        exact ⟨x^2, by ring⟩
      have : 3*x^2 + 1 ≡ 0 + 1 [ZMOD 3] := Int.ModEq.add_right 1 this
      simpa
    rw [l7]
    exact ⟨this, h_dvd, h_ge⟩
  have l9 : ∀ (x y : ℤ), 3*x^2 + 1 = 13 → (3*x^2 + 1)*(y^2 - 10) = 507 → x^2 = 4 ∧ y^2 = 49 := by
    intro x y hx h
    have : y^2 - 10 = 39 := by
      linarith
    constructor
    · linarith [hx]
    · linarith
  have l10 : ∀ (x y : ℤ), 3*x^2 + 1 = 1 → (3*x^2 + 1)*(y^2 - 10) = 507 → x = 0 ∧ y^2 = 517 ∧ ¬∃ (k : ℤ), k^2 = 517 := by
    intro x y hx h
    have : y^2 - 10 = 507 := by
      linarith
    constructor
    · nlinarith
    constructor
    · linarith
    · intro ⟨k, hk⟩
      have : k^2 = 517 := hk
      have : 517 < 23^2 := by norm_num
      have : 517 > 22^2 := by norm_num
      have : k ≤ 22 ∨ k ≥ 23 := by omega
      rcases this with (h | h)
      · have : k ≤ 22 := h
        have : k^2 ≤ 22^2 := by nlinarith
        nlinarith
      · have : k ≥ 23 := h
        have : k^2 ≥ 23^2 := by nlinarith
        nlinarith
  have l11 : ∀ (x : ℤ), 3*x^2 + 1 = 169 → ¬∃ (k : ℤ), k^2 = 56 := by
    intro x hx
    intro ⟨k, hk⟩
    have : x^2 = 56 := by linarith
    have : 56 < 8^2 := by norm_num
    have : 56 > 7^2 := by norm_num
    have : k ≤ 7 ∨ k ≥ 8 := by omega
    rcases this with (h | h)
    · have : k ≤ 7 := h
      have : k^2 ≤ 7^2 := by nlinarith
      nlinarith
    · have : k ≥ 8 := h
      have : k^2 ≥ 8^2 := by nlinarith
      nlinarith
  have ts_1 : ∃ (x y : ℤ), 3*x^2*y^2 = 588 := by
    rcases tc_1 with ⟨x, y, h⟩
    have h1 : (3*x^2 + 1)*(y^2 - 10) = 507 := l1 x y h
    have h3 : 3*x^2 + 1 ∣ 507 ∧ 3*x^2 + 1 ≥ 1 := l3 x y h1
    have h8 : 3*x^2 + 1 ∈ ({1, 13, 169} : Set ℤ) := l8 x h3
    simp at h8
    rcases h8 with (h8 | h8 | h8)
    · have : 3*x^2 + 1 = 1 := h8
      have : x = 0 ∧ y^2 = 517 ∧ ¬∃ (k : ℤ), k^2 = 517 := l10 x y this h1
      rcases this with ⟨hx, hy, h_no_sq⟩
      exfalso
      apply h_no_sq
      exact ⟨y, by rw [hy]⟩
    · have : 3*x^2 + 1 = 13 := h8
      have : x^2 = 4 ∧ y^2 = 49 := l9 x y this h1
      rcases this with ⟨hx, hy⟩
      refine ⟨x, y, ?_⟩
      nlinarith
    · have : 3*x^2 + 1 = 169 := h8
      have : ¬∃ (k : ℤ), k^2 = 56 := l11 x this
      have : x^2 = 56 := by linarith
      exfalso
      apply this
      exact ⟨x, this⟩
  trivial