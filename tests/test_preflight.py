"""GPU arch preflight — "이미지 안 실물이 이 GPU 에서 도나"를 학습 전에 판정한다.

이 판정을 틀리면 두 방향으로 비싸다: 통과시켜야 할 걸 막으면 런이 아예 안 돌고,
막아야 할 걸 통과시키면 12분 뒤 원인 불명 CUDA 에러로 죽는다(2026-07-28 실측).
"""

from training_framework_comparison_tutorial.trainers._preflight import arch_supported


def test_native_kernel_present():
    """장치 arch 가 목록에 그대로 있으면 통과."""
    assert arch_supported((8, 9), ["sm_80", "sm_86", "sm_89", "sm_90"])


def test_blackwell_on_cu124_torch_is_blocked():
    """2026-07-28 실제 사고: torch 2.6(cu12.4)에 sm_120 이 없다 → 막아야 한다."""
    assert not arch_supported((12, 0), ["sm_50", "sm_80", "sm_86", "sm_90", "sm_90a"])


def test_ptx_allows_forward_jit():
    """compute_XX(PTX)가 장치보다 낮거나 같으면 JIT 로 돌 수 있다 → 막지 않는다."""
    assert arch_supported((12, 0), ["sm_90", "compute_90"])


def test_ptx_newer_than_device_does_not_help():
    """장치보다 높은 PTX 는 역호환이 안 된다 → 통과시키면 안 된다."""
    assert not arch_supported((7, 5), ["sm_90", "compute_90"])


def test_malformed_entries_are_ignored():
    """arch 목록에 예상 밖 항목이 섞여도 죽지 않는다(판정만 보수적으로)."""
    assert not arch_supported((12, 0), ["sm_", "weird", "compute_x"])
    assert arch_supported((9, 0), ["junk", "sm_90"])
