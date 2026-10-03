theorem algebra_2varlineareq_fp3zeq11_3tfm1m5zeqn68_feqn10_zeq7 : True := by
  have tc_1 : ℤ → ℤ → Prop := λ f z => f + 3 * z = 11
  have tc_2 : ℤ → ℤ → Prop := λ f z => 3 * (f - 1) - 5 * z = -68
  have l1 : ℤ → ℤ → Prop := λ f z => 3 * f - 5 * z = -65
  have h_l1_from_tc_2 : ∀ f z, tc_2 f z → l1 f z := by
    intro f z h_tc2
    dsimp [l1, tc_2] at *
    linarith
  have l2 : ℤ → ℤ → Prop := λ f z => f = 11 - 3 * z
  have h_l2_from_tc_1 : ∀ f z, tc_1 f z → l2 f z := by
    intro f z h_tc1
    dsimp [l2, tc_1] at *
    linarith
  have l3 : ℤ → ℤ → Prop := λ f z => 3 * f + 9 * z = 33
  have h_l3_from_tc_1 : ∀ f z, tc_1 f z → l3 f z := by
    intro f z h_tc1
    dsimp [l3, tc_1] at *
    linarith
  have l4 : ℤ → ℤ → Prop := λ f z => z = 7
  have h_l4_from_l1_l3 : ∀ f z, l1 f z → l3 f z → l4 f z := by
    intro f z h_l1 h_l3
    dsimp [l4, l1, l3] at *
    linarith
  have l5 : ℤ → ℤ → Prop := λ f z => f = -10
  have h_l5_from_l2_l4 : ∀ f z, l2 f z → l4 f z → l5 f z := by
    intro f z h_l2 h_l4
    dsimp [l5, l2, l4] at *
    linarith
  have l6 : ℤ → ℤ → Prop := λ f z => 3 * ((11 - 3 * z) - 1) - 5 * z = -68
  have h_l6_from_l2_tc_2 : ∀ f z, l2 f z → tc_2 f z → l6 f z := by
    intro f z h_l2 h_tc2
    dsimp [l6, l2, tc_2] at *
    linarith
  have l7 : ℤ → ℤ → Prop := λ f z => 3 * f - 3 - 5 * z = -68 ∧ f = 11 - 3 * z
  have h_l7_from_tc_2_l2 : ∀ f z, tc_2 f z → l2 f z → l7 f z := by
    intro f z h_tc2 h_l2
    dsimp [l7, l2, tc_2] at *
    constructor <;> linarith
  have l8 : ℤ → ℤ → Prop := λ f z => 30 - 14 * z = -68
  have h_l8_from_l6 : ∀ f z, l6 f z → l8 f z := by
    intro f z h_l6
    dsimp [l8, l6] at *
    linarith
  have l9 : ℤ → ℤ → Prop := λ f z => z = 7
  have h_l9_from_l8 : ∀ f z, l8 f z → l9 f z := by
    intro f z h_l8
    dsimp [l9, l8] at *
    linarith
  have l10 : ℤ → ℤ → Prop := λ f z => f = -10
  have h_l10_from_l2_l9 : ∀ f z, l2 f z → l9 f z → l10 f z := by
    intro f z h_l2 h_l9
    dsimp [l10, l2, l9] at *
    linarith
  have l11 : ℤ → ℤ → Prop := λ f z => 3 * ((11 - 3 * z) - 1) - 5 * z = -68
  have h_l11_from_l2_tc_2 : ∀ f z, l2 f z → tc_2 f z → l11 f z := by
    intro f z h_l2 h_tc2
    dsimp [l11, l2, tc_2] at *
    linarith
  have l12 : ℤ → ℤ → Prop := λ f z => 3 * f - 5 * z = -65
  have h_l12_from_tc_2 : ∀ f z, tc_2 f z → l12 f z := by
    intro f z h_tc2
    dsimp [l12, tc_2] at *
    linarith
  have l13 : ℤ → ℤ → Prop := λ f z => 3 * (11 - 3 * z) - 5 * z = -65
  have h_l13_from_l2_l12 : ∀ f z, l2 f z → l12 f z → l13 f z := by
    intro f z h_l2 h_l12
    dsimp [l13, l2, l12] at *
    linarith
  have l14 : ℤ → ℤ → Prop := λ f z => z = 7
  have h_l14_from_l13 : ∀ f z, l13 f z → l14 f z := by
    intro f z h_l13
    dsimp [l14, l13] at *
    linarith
  have l15 : ℤ → ℤ → Prop := λ f z => f = -10
  have h_l15_from_l2_l14 : ∀ f z, l2 f z → l14 f z → l15 f z := by
    intro f z h_l2 h_l14
    dsimp [l15, l2, l14] at *
    linarith
  have l16 : ℤ → ℤ → Prop := λ f z => 14 * z = 98
  have h_l16_from_l1_l3 : ∀ f z, l1 f z → l3 f z → l16 f z := by
    intro f z h_l1 h_l3
    dsimp [l16, l1, l3] at *
    linarith
  have l17 : ℤ → ℤ → Prop := λ f z => 3 * (11 - 3 * z) - 5 * z = -65
  have h_l17_from_l2_l12 : ∀ f z, l2 f z → l12 f z → l17 f z := by
    intro f z h_l2 h_l12
    dsimp [l17, l2, l12] at *
    linarith
  have l18 : ℤ → ℤ → Prop := λ f z => z = 7
  have h_l18_from_l1_l3 : ∀ f z, l1 f z → l3 f z → l18 f z := by
    intro f z h_l1 h_l3
    dsimp [l18, l1, l3] at *
    linarith
  have l19 : ℤ → ℤ → Prop := λ f z => f = -10
  have h_l19_from_l2_l18 : ∀ f z, l2 f z → l18 f z → l19 f z := by
    intro f z h_l2 h_l18
    dsimp [l19, l2, l18] at *
    linarith
  have l20 : ℤ → ℤ → Prop := λ f z => 3 * (11 - 3 * z) - 5 * z = -65
  have h_l20_from_l2_l12 : ∀ f z, l2 f z → l12 f z → l20 f z := by
    intro f z h_l2 h_l12
    dsimp [l20, l2, l12] at *
    linarith
  have l21 : ℤ → ℤ → Prop := λ f z => 33 - 9 * z - 5 * z = -65
  have h_l21_from_l13 : ∀ f z, l13 f z → l21 f z := by
    intro f z h_l13
    dsimp [l21, l13] at *
    linarith
  have ts_1 : ℤ → ℤ → Prop := λ f z => f = -10 ∧ z = 7
  have h_ts_1 : ∀ f z, tc_1 f z → tc_2 f z → ts_1 f z := by
    intro f z h_tc1 h_tc2
    have h_l2 := h_l2_from_tc_1 f z h_tc1
    have h_l12 := h_l12_from_tc_2 f z h_tc2
    have h_l13 := h_l13_from_l2_l12 f z h_l2 h_l12
    have h_l14 := h_l14_from_l13 f z h_l13
    have h_l15 := h_l15_from_l2_l14 f z h_l2 h_l14
    exact ⟨h_l15, h_l14⟩
  trivial