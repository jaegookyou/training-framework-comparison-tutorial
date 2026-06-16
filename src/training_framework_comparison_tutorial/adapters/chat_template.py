"""캐논 학습용 chat template.

왜 필요한가: 통제비교 대상 모델이 **base 모델**(Qwen3-8B-Base)이라 그 토크나이저의
기본 template 에는 `assistant_only_loss` 가 요구하는 `{% generation %}` 마커가 없다
(base 의 ChatML template 에 generation 마커만 빠져 있음). 그러면 TRL 이 assistant
토큰 마스크를 못 뽑아 깨진다.

해결: reasoning SFT 트랙용 **단순 ChatML 학습 template 한 장**을 코드에 캐논으로 정의하고
모든 프레임워크가 같은 걸 쓰게 한다(데이터·포맷 동일 = 통제비교). assistant 응답을
`{% generation %}...{% endgeneration %}` 로 감싸 reasoning(`<think>` 인라인) 전체에만
loss 가 걸리게 한다. base 모델이라 맞출 기존 instruct 동작이 없으니 우리가 포맷을 정의한다.

NOTE: transformers 는 chat template 을 trim_blocks/lstrip_blocks 로 컴파일하므로
`{%- ... %}`(블록) + `{{- ... }}`(출력) 패턴이면 들여쓰기·개행이 출력에 새지 않는다.
generation 블록은 **assistant 의 학습 대상 토큰**(content + 종료 `<|im_end|>`)만 감싼다.
role 헤더 `<|im_start|>assistant\n` 는 추론 시 add_generation_prompt 가 주는 프롬프트
큐라 마스크 밖에 둔다(학습/추론 정합).
"""

from __future__ import annotations

# reasoning SFT: assistant content(<think> CoT + 답)에만 loss. 멀티모달/tool 미사용.
REASONING_CHATML = """\
{%- for message in messages %}
    {%- if message['role'] == 'assistant' %}
        {{- '<|im_start|>assistant\n' }}
        {%- generation %}
        {{- message['content'] + '<|im_end|>\n' }}
        {%- endgeneration %}
    {%- else %}
        {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"""

# 프레임워크/트랙 추가 시 여기에 등록.
CHAT_TEMPLATES: dict[str, str] = {
    "reasoning_chatml": REASONING_CHATML,
}


def resolve_chat_template(name: str | None) -> str | None:
    """config 의 model.chat_template 이름 → template 문자열.

    None/빈 값이면 None(토크나이저 자기 template 유지). 알 수 없는 이름은 에러.
    """
    if not name:
        return None
    try:
        return CHAT_TEMPLATES[name]
    except KeyError:
        raise ValueError(f"unknown chat_template: {name!r}") from None
