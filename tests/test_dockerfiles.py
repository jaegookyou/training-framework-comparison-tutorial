"""Dockerfile 최소 문법 검증 — 빌드를 돌리기 전에 로컬에서 잡는다.

왜 있나(2026-07-28): Dockerfile 주석을 스크립트로 삽입하다 **주석 안의 "FROM base 불가" 문구를
쪼개** 그 잔해가 명령 라인이 되어버렸다. Docker 는 `FROM requires either one or three arguments`
로 거절했고, 그 사실을 **CI 에서 30분 뒤에야** 알았다(megatron 2개 빌드 낭비).
로컬에서 0.01초에 잡히는 종류의 실수라 여기 둔다.
"""

import re
from pathlib import Path

import pytest

DOCKERFILES = sorted((Path(__file__).resolve().parents[1] / "docker").glob("*.Dockerfile"))

# Docker 명령어(우리가 쓰는 것 + 표준). 줄 시작이 이 중 하나면 명령 라인으로 본다.
INSTRUCTIONS = {
    "FROM", "RUN", "ARG", "ENV", "COPY", "ADD", "LABEL", "WORKDIR",
    "ENTRYPOINT", "CMD", "EXPOSE", "USER", "VOLUME", "SHELL", "HEALTHCHECK", "ONBUILD",
}


def test_dockerfile_set_is_not_empty():
    assert DOCKERFILES, "docker/*.Dockerfile 이 하나도 없다"


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda p: p.stem)
def test_no_stray_instruction_like_lines(path):
    """명령처럼 보이는 줄은 전부 **유효한 명령**이어야 한다.

    주석을 잘못 쪼개면 문장 조각이 명령 라인 자리에 남는데, 그게 정확히 이 테스트가 잡는 것이다.
    """
    continuation = False
    heredoc = None                             # 열려 있는 heredoc 종료 토큰
    for lineno, raw in enumerate(path.read_text().split("\n"), 1):
        # RUN 안의 heredoc 본문(셸 `if ...`, 파이썬 `import ...`)은 명령 자리가 아니다.
        # 이걸 안 보면 정상 Dockerfile 이 무더기로 걸린다(2026-07-28 첫 판에서 오탐 2건).
        if heredoc is not None:
            if raw.strip() == heredoc:
                heredoc = None
            continue
        opened = re.search(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", raw)

        if continuation:                       # 앞 줄이 `\` 로 이어진 인자 줄 — 명령 위치가 아니다
            continuation = raw.rstrip().endswith("\\")
            if opened:
                heredoc = opened.group(1)
            continue
        continuation = raw.rstrip().endswith("\\")
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if opened:
            heredoc = opened.group(1)
        head = line.split(maxsplit=1)[0].upper()
        assert head in INSTRUCTIONS, (
            f"{path.name}:{lineno} 가 명령 라인 자리인데 유효한 Docker 명령이 아니다: {raw!r}\n"
            f"  (주석이 쪼개져 문장 조각이 남았을 가능성이 높다)"
        )


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda p: p.stem)
def test_from_has_valid_arity(path):
    """`FROM` 은 인자가 1개(이미지) 또는 3개(이미지 AS 별칭)여야 한다 — CI 가 거절한 그 규칙."""
    for lineno, raw in enumerate(path.read_text().split("\n"), 1):
        if not re.match(r"^FROM\s", raw):
            continue
        parts = raw.split()[1:]
        assert len(parts) in (1, 3), f"{path.name}:{lineno} FROM 인자 수가 {len(parts)}: {raw!r}"
        if len(parts) == 3:
            assert parts[1].upper() == "AS", f"{path.name}:{lineno} FROM ... AS ... 형식이 아니다"
