from src.physiology import monitor_drift
def test_drift_monitor_flags_sustained_large_residuals():
    assert monitor_drift([25,30,28,29,31])["drift_detected"]
