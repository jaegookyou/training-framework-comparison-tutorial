"""W&B run 식별을 **프레임워크 무관하게** env 한 곳에서 주입한다.

왜 env 인가: 8 프레임워크가 wandb 를 부르는 방식이 제각각이다(HF 콜백 / torchtitan WandBLogger /
verl tracking / slime init_wandb_primary / megatron global_vars). 그런데 **여섯 곳
어디도 wandb.init 에 group·tags 를 넘기지 않는다** → wandb 가 env 에서 읽어 채운다. 즉 env 로 주면
프레임워크별 코드 0 줄로 같은 식별자가 붙는다(반대로 프레임워크마다 배선하면 6 곳이 어긋난다).

확인한 사실(2026-07-28, upstream 소스 실물 — 추정 아님):
  - `wandb/sdk/wandb_settings.py` update_from_env_vars: `WANDB_` prefix 를 떼고 소문자 필드로 매핑
    (WANDB_NAME→run_name, WANDB_RUN_GROUP→run_group, WANDB_PROJECT→project), **WANDB_TAGS 는
    콤마로 split** 된다.
  - `wandb/sdk/wandb_init.py:1405-1422` `if tags is not None:` — init 인자가 None 이면 env 를
    **덮어쓰지 않는다** → 프레임워크가 project/name 만 넘겨도 group·tags 는 우리 env 가 산다.
  - `transformers/integrations/integration_utils.py:763`
    `project=os.getenv("WANDB_PROJECT", "huggingface")` → trl·unsloth 는 이 env 가 없으면 개인
    `huggingface` 프로젝트로 샌다. configs 의 `wandb.project` 가 죽은 설정이던 원인이 이것.
  - `torchtitan/components/metrics.py:149-160` WandBLogger 는 표준 WANDB_PROJECT 를 읽지만 tags 는
    **비표준 WANDB_RUN_TAGS** 를 `tags=` 로 그대로 넘긴다 → 문자열이면 init 의 `tuple(tags)` 가
    **글자 단위로 쪼갠다**. 그래서 WANDB_RUN_TAGS 는 **절대 세팅하지 않는다**(안 주면 None →
    표준 WANDB_TAGS 경로로 정상 동작).

**예외 1 — slime**: `--wandb-group` 하나가 group 과 run name 을 동시에 정한다(wandb_utils.py:45-50).
group=framework 로 맞추면 run 이름이 전부 "slime" 이 되어 못 알아본다 → slime 만 group=run_name 을
유지한다. framework 로 묶어 보는 건 태그로 성립하므로 손해가 없다.
**예외 2 — project**: 프레임워크가 명시로 넘기는 값(verl trainer.project_name, slime/megatron
--wandb-project)이 env 를 이긴다 → 그 값도 같은 `project()` 에서
뽑아 써서 한 곳으로 수렴시킨다.
"""

from __future__ import annotations

import os

from ..config import RunConfig
from ._dist import MULTINODE_MECHANISM

# 스모크(=debug.max_steps>0) 전용 프로젝트. 스모크는 5 step 짜리 진단용 쓰레기 run 이라 실측
# 프로젝트(tfct-sft 등)에 섞이면 가로비교 화면이 오염된다(W&B run 삭제는 수동이라 청소도 비싸다).
SMOKE_PROJECT = "tfct-smoke"


def tier(nodes: int, gpus: int) -> str:
    """검증 티어 이름. 우리가 런을 나눠 보는 단위(SNSG 스윕 / MNMG 배선검증)와 같은 눈금."""
    if nodes > 1:
        return "mnmg"
    return "snmg" if gpus > 1 else "snsg"


def project(cfg: RunConfig) -> str:
    """이 run 이 갈 W&B 프로젝트. 스모크면 격리 프로젝트로 강제한다.

    프로젝트 경계 = **겹쳐 보고 싶은 단위** → method 별(configs/*/_base.yaml 의 wandb.project).
    framework 로 쪼개지 않는 이유: 프레임워크가 유일한 독립변수라 쪼개면 가로 통제비교(겹쳐보기)가
    구조적으로 불가능해진다. 프레임워크별로 나눠 보는 건 group·tag 로 얻는다.
    """
    if cfg.is_smoke():
        return SMOKE_PROJECT
    return cfg.section("wandb").get("project", f"tfct-{cfg.method}")


def tags(cfg: RunConfig) -> list[str]:
    """run 에 붙일 태그. W&B 에서 이 축들로 필터/그룹을 만든다."""
    scale = cfg.section("scale")
    nodes = int(scale.get("nodes", 1))
    gpus = int(scale.get("gpus", 1))
    out = [
        cfg.method,
        cfg.framework,
        cfg.tuning,
        tier(nodes, gpus),
        f"nodes-{nodes}",
        f"gpus-{gpus}",
    ]
    if cfg.is_smoke():
        out.append("smoke")
    # 멀티노드는 "어떤 메커니즘이 검증됐나"가 정보의 단위다(조합이 아니라).
    mech = MULTINODE_MECHANISM.get((cfg.method, cfg.framework))
    if nodes > 1 and mech:
        out.append(f"mech-{mech}")
    return out


def env(cfg: RunConfig) -> dict[str, str]:
    """이 run 이 필요로 하는 WANDB_* 매핑(순수 함수 — 테스트가 여기를 본다)."""
    return {
        "WANDB_PROJECT": project(cfg),
        "WANDB_NAME": cfg.run_name(),
        # group=framework: 겹쳐보기(프로젝트)와 나눠보기(그룹)를 동시에 성립시키는 경계.
        "WANDB_RUN_GROUP": cfg.framework,
        "WANDB_TAGS": ",".join(tags(cfg)),
    }


def apply_env(cfg: RunConfig) -> dict[str, str]:
    """WANDB_* 를 프로세스 env 에 주입한다(dispatch 초입 = 전 프레임워크 공통 통로).

    `setdefault` 인 이유: 런치 때 `--env WANDB_TAGS=...` 로 임시 라벨을 붙이는 일이 있는데,
    그때 config 파생값이 사람이 준 값을 덮으면 안 된다(명시 > 파생).
    """
    applied = env(cfg)
    for key, value in applied.items():
        os.environ.setdefault(key, value)
    return applied
