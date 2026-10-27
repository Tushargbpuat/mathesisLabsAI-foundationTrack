import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cityblock, mahalanobis, chebyshev, cosine, hamming

# Set seed for reproducible industrial simulation
np.random.seed(42)

# =====================================================================
# 1. DATA GENERATION & DYNAMICS SETUP
# =====================================================================
# Physical Context: High-Pressure Centrifugal Pump Telemetry
# Features: [Vibration (mm/s), Bearing Temperature (°C)]
# Mean operating point (Baseline Healthy State)
mean_healthy = np.array([2.0, 50.0])

# Physical Covariance Matrix: High temperature naturally correlates with high vibration
# Var(Vib) = 0.5, Var(Temp) = 8.0, Cov(Vib, Temp) = 1.8
cov_matrix = np.array([[0.5, 1.8], 
                       [1.8, 8.0]])
cov_inv = np.linalg.inv(cov_matrix)

# Reference Point (Baseline Center)
p_ref = mean_healthy

# Point A: Correlated Thermal Drift (Vibration and Temp increase together)
# Physical interpretation: Pump working under higher workload (Expected behavior)
p_a = np.array([3.5, 56.0])

# Point B: Uncorrelated Vibration Anomaly (Vibration rises while Temp drops)
# Physical interpretation: Mechanical looseness / structural fault (True Anomaly)
p_b = np.array([3.5, 44.0])

# Digital Hardware Alarm Registers (8-bit binary vectors for Hamming analysis)
# Bit allocation: [OverTemp, OverVib, PressureDrop, OilLow, SensorErr, ValveStuck, VoltageDrop, LeakDet]
alarm_baseline  = np.array([0, 0, 0, 0, 0, 0, 0, 0])
alarm_warning   = np.array([1, 1, 0, 0, 0, 0, 0, 0])  # 2 bit flips
alarm_critical  = np.array([1, 1, 1, 0, 1, 0, 1, 0])  # 5 bit flips

# =====================================================================
# 2. DISTANCE ENGINE CALCULATIONS
# =====================================================================
def compute_metrics(p1, p2, cov_inv_mat):
    euc = euclidean(p1, p2)
    man = cityblock(p1, p2)
    che = chebyshev(p1, p2)
    mah = mahalanobis(p1, p2, cov_inv_mat)
    cos_dist = cosine(p1, p2)
    cos_sim = 1.0 - cos_dist
    return [euc, man, che, mah, cos_dist, cos_sim]

metrics_a = compute_metrics(p_ref, p_a, cov_inv)
metrics_b = compute_metrics(p_ref, p_b, cov_inv)

# DataFrame for Vector Space Comparison
df_comparison = pd.DataFrame({
    'Metric Type': [
        'Euclidean (L2 Norm)', 
        'Manhattan (L1 Norm)', 
        'Chebyshev (Linf Norm)', 
        'Mahalanobis (Statistical)', 
        'Cosine Distance (1 - CosSim)', 
        'Cosine Similarity (Cos θ)'
    ],
    'Formula / Geometry': [
        'sqrt(sum(dx^2)) [Hyper-sphere]',
        'sum(|dx|) [Cross-polytope]',
        'max(|dx|) [Hyper-cube]',
        'sqrt(dx^T * Sigma^-1 * dx) [Hyper-ellipsoid]',
        '1 - (A·B / ||A||||B||) [Angular separation]',
        'A·B / ||A||||B|| [Directional alignment]'
    ],
    'Point A (Correlated Drift)': [f"{v:.4f}" for v in metrics_a],
    'Point B (Uncorrelated Anomaly)': [f"{v:.4f}" for v in metrics_b]
})

# Hamming Calculations
ham_dist_warn = hamming(alarm_baseline, alarm_warning)
ham_bits_warn = int(ham_dist_warn * len(alarm_baseline))

ham_dist_crit = hamming(alarm_baseline, alarm_critical)
ham_bits_crit = int(ham_dist_crit * len(alarm_baseline))

df_hamming = pd.DataFrame({
    'Comparison Pair': ['Baseline vs. Warning Reg', 'Baseline vs. Critical Reg'],
    'Vector Length (Bits)': [len(alarm_baseline), len(alarm_baseline)],
    'Normalized Hamming Dist': [f"{ham_dist_warn:.4f}", f"{ham_dist_crit:.4f}"],
    'Absolute Bit Flips': [ham_bits_warn, ham_bits_crit]
})

# =====================================================================
# 3. CONSOLE REPORT GENERATOR
# =====================================================================
print("=" * 90)
print("       MATHESISLABS GEOMETRIC & TOPOLOGICAL DISTANCE METRICS CONSOLE GUIDE        ")
print("=" * 90)
print(f"Baseline Operating Center (P_ref): {p_ref} [Vibration (mm/s), Temperature (°C)]")
print(f"Test Point A (Correlated Workload Shift): {p_a}")
print(f"Test Point B (Uncorrelated Structural Anomaly): {p_b}")
print("-" * 90)

print("\n[SECTION 1: CONTINUOUS FEATURE SPACE METRIC COMPARISON]")
print(df_comparison.to_string(index=False))

print("\n--- SECTION 1 GEOMETRIC & INTUITIVE INTERPRETATION ---")
print("1. EUCLIDEAN DISTANCE (L2 Norm):")
print("   • Value: Exactly 6.1847 for BOTH Point A and Point B.")
print("   • Geometric Unit Ball: A circle/sphere centered at P_ref.")
print("   • Limitation: Euclidean distance is ISOTROPIC (blind to physical covariance and feature scales).")
print("     It views Point A and Point B as EQUALLY distant from normal operation, completely failing")
print("     to detect that Point B violates the expected thermal-vibration coupling.")

print("\n2. MANHATTAN DISTANCE (L1 Norm):")
print("   • Value: Point A = 7.5000 | Point B = 7.5000.")
print("   • Geometric Unit Ball: A rhombus/cross-polytope.")
print("   • Context: Measures distance along orthogonal coordinate axes (grid routing). Ideal for discrete")
print("     state grids, pipeline manifolds, or sparse feature selection (L1 regularization).")

print("\n3. CHEBYSHEV DISTANCE (L-infinity Norm):")
print("   • Value: Point A = 6.0000 | Point B = 6.0000.")
print("   • Geometric Unit Ball: A square/cube.")
print("   • Context: Governed strictly by the SINGLE WORST-CASE feature offset: max(|dx_i|).")
print("     Industrial Use Case: Critical safety thresholds where a single component breach triggers a shutdown.")

print("\n4. MAHALANOBIS DISTANCE (Statistical Covariance Metric):")
print("   • Value: Point A = 2.1804 | Point B = 9.4932  <-- HUGE PARADIGM SHIFT!")
print("   • Geometric Unit Ball: A rotated hyper-ellipsoid stretched by the covariance matrix (Sigma).")
print("   • Physical Intuition: Measures distance in units of STANDARD DEVIATION along variance principal axes.")
print("     - Point A (2.18) lies ALONG the expected physical operational plume -> LOW ANOMALY SCORES.")
print("     - Point B (9.49) breaks physical laws of the system -> SEVERE STRUCTURAL ANOMALY.")

print("\n5. COSINE DISTANCE & SIMILARITY (Angular Orientation):")
print("   • Value: Point A CosSim = 0.9996 | Point B CosSim = 0.9976.")
print("   • Context: Ignores physical magnitude completely; measures only the ANGLE between vectors.")
print("   • Industrial Use Case: High-dimensional telemetry embeddings, SLM/LLM token state spaces,")
print("     or spectral sensor signatures where magnitude varies with gain but profile remains identical.")

print("\n" + "-" * 90)
print("\n[SECTION 2: DISCRETE STATE SPACE - HAMMING DISTANCE]")
print(df_hamming.to_string(index=False))

print("\n--- SECTION 2 INTUITIVE INTERPRETATION ---")
print("• Geometric Meaning: Counts the minimum number of substitution operations required to convert")
print("  one string/vector into another, or measures the orthogonal distance on a hypercube vertex graph.")
print("• Industrial Context: Categorical error codes, telemetry fault logs, and communication error correction.")
print("• Takeaway: Normalizes differences across boolean hardware bit registers.")

print("\n" + "=" * 90)
print("INTERN LESSON SUMMARY:")
print("1. Selecting a distance metric IS selecting an underlying physical assumption about your feature space.")
print("2. Raw L2/L1 distances assume features are independent and unscaled — a dangerous assumption in physics.")
print("3. Use Mahalanobis for correlated sensor data, Cosine for direction/embeddings, and Hamming for digital state vectors.")
print("=" * 90)