# Joint-limit metric correction

The original evaluator mislabeled the inherited 90% soft training margin as the physical joint limit. Corrected metrics use the validated URDF limits and retain soft-margin exceedance separately. Loop 1 has no physical joint-limit violation. Its REJECT decision remains warranted by pivot translation and stalled backward turns. The sealed original files remain unchanged; use these corrected metrics for comparisons. Subsequent evaluations use the corrected calculation.
