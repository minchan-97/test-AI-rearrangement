"""
logit_lens_app.py — LLM 내부를 선형 투영으로 들여다보기
========================================================
탐구 질문:
  "비선형 함수를 선형으로 재정렬하면, 어떤 분기로 설명이 이어졌는지 알 수 있나?"

이 도구가 하는 일 (logit lens):
  트랜스포머는 각 층에서 residual stream(잔차 흐름)이라는 벡터를 갱신한다.
  최종 층의 벡터에 unembedding 행렬을 곱하면 다음 토큰 확률이 나온다.
  → 그렇다면 **중간 층의 벡터에도 같은 걸 곱해보면?**
     그 층에서 모델이 "무슨 토큰을 생각 중이었는지" 읽을 수 있다.

  이것이 선형 재정렬이다. 비선형(attention softmax, GELU, LayerNorm)을
  거친 중간 상태를, 최종 출력용 선형 사상으로 억지로 읽는 것.
  → 오차가 있다. 특히 초기 층에서 크다. 그 오차 자체가 관찰 대상이다.

실행:
    pip install -r requirements_lens.txt
    streamlit run logit_lens_app.py

주의: 처음 실행 시 모델을 내려받는다(gpt2 약 500MB). 인터넷 필요.
      CPU로도 동작하지만 느리다.
"""
import streamlit as st

st.set_page_config(page_title="Logit Lens 탐구", layout="wide")

# ── 의존성 확인 ────────────────────────────────
try:
    import torch
    import numpy as np
    import plotly.graph_objects as go
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    DEPS_OK = True
    DEP_ERR = ""
except Exception as e:
    DEPS_OK = False
    DEP_ERR = str(e)

if not DEPS_OK:
    st.error(f"의존성이 없습니다: {DEP_ERR}")
    st.code("pip install torch transformers plotly streamlit numpy")
    st.stop()


# ── 모델 로드 ──────────────────────────────────
@st.cache_resource
def load_model(name="gpt2"):
    tok = GPT2Tokenizer.from_pretrained(name)
    model = GPT2LMHeadModel.from_pretrained(name, output_hidden_states=True)
    model.eval()
    return tok, model


# ── 핵심: 층별 선형 투영 ────────────────────────
def logit_lens(model, tok, text, apply_final_ln=True, topk=5):
    """각 층의 residual stream을 unembedding에 선형 투영.

    Returns: layers x [(token, prob), ...]  마지막 위치 기준
    """
    ids = tok.encode(text, return_tensors="pt")
    with torch.no_grad():
        out = model(ids)
    # hidden_states: (embedding, layer1, ..., layerN) 각 (1, seq, d_model)
    hs = out.hidden_states
    W_U = model.lm_head.weight            # (vocab, d_model)
    ln_f = model.transformer.ln_f         # 최종 LayerNorm

    results = []
    for li, h in enumerate(hs):
        v = h[0, -1, :]                   # 마지막 토큰 위치의 residual
        if apply_final_ln:
            # 최종 LN을 적용해야 unembedding 스케일과 맞음.
            # 이걸 끄면 초기 층이 특히 엉망이 된다 (관찰 포인트).
            v = ln_f(v)
        logits = W_U @ v                  # 선형 투영
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, topk)
        results.append([
            (tok.decode([int(i)]), float(p))
            for i, p in zip(top.indices, top.values)
        ])
    return results, ids


def track_token_prob(model, tok, text, target_token, apply_final_ln=True):
    """특정 토큰의 확률이 층을 지나며 어떻게 변하는지."""
    ids = tok.encode(text, return_tensors="pt")
    tgt_ids = tok.encode(target_token)
    if not tgt_ids:
        return None, None
    tgt = tgt_ids[0]
    with torch.no_grad():
        out = model(ids)
    hs = out.hidden_states
    W_U = model.lm_head.weight
    ln_f = model.transformer.ln_f
    probs = []
    for h in hs:
        v = h[0, -1, :]
        if apply_final_ln:
            v = ln_f(v)
        p = torch.softmax(W_U @ v, dim=-1)
        probs.append(float(p[tgt]))
    return probs, tok.decode([tgt])


# ── UI ─────────────────────────────────────────
st.title("Logit Lens — LLM 내부를 선형으로 들여다보기")
st.caption("비선형 연산을 거친 중간 층을, 최종 출력용 선형 사상으로 억지로 읽으면 뭐가 보일까?")

with st.spinner("모델 로딩 중... (처음엔 다운로드로 시간이 걸립니다)"):
    tok, model = load_model()
n_layers = model.config.n_layer
st.success(f"gpt2 로드 완료 — {n_layers}개 층, d_model={model.config.n_embd}")

tab1, tab2, tab3 = st.tabs(["① 층별 예측 보기", "② 특정 토큰 추적", "③ 선형화의 한계"])

# ── 탭1 ──
with tab1:
    st.subheader("각 층에서 모델은 무슨 토큰을 생각 중이었나")
    text = st.text_input("입력 문장", "The Eiffel Tower is located in the city of",
                         key="t1")
    apply_ln = st.checkbox("최종 LayerNorm 적용 (끄면 초기 층이 망가지는 걸 볼 수 있음)",
                           value=True, key="ln1")
    topk = st.slider("각 층에서 볼 상위 토큰 수", 1, 10, 5)

    if st.button("들여다보기", type="primary", key="b1"):
        with st.spinner("순전파 + 층별 투영 중..."):
            res, ids = logit_lens(model, tok, text, apply_ln, topk)

        st.write(f"입력 토큰: `{tok.decode(ids[0])}`")
        st.markdown("**아래로 갈수록 상위 층** — 예측이 어떻게 변해가나")

        rows = []
        for li, top in enumerate(res):
            label = "embed" if li == 0 else f"layer {li}"
            cells = " | ".join(f"{t.strip()!r} {p:.1%}" for t, p in top)
            rows.append({"층": label, "상위 예측": cells})
        st.dataframe(rows, use_container_width=True, height=520)

        st.info(
            "관찰 포인트: 초기 층은 대체로 의미 없는 토큰. "
            "중간 이후 어느 지점에서 답이 갑자기 떠오르고, 상위 층에서 굳어진다. "
            "이 '떠오르는 지점'이 층별로 연속적이지, 이산적 분기가 아니라는 게 핵심."
        )

# ── 탭2 ──
with tab2:
    st.subheader("특정 토큰의 확률이 층을 지나며 어떻게 자라나")
    text2 = st.text_input("입력 문장", "The Eiffel Tower is located in the city of",
                          key="t2")
    target = st.text_input("추적할 토큰 (앞 공백 포함 주의)", " Paris", key="t2b")
    apply_ln2 = st.checkbox("최종 LayerNorm 적용", value=True, key="ln2")

    if st.button("추적", type="primary", key="b2"):
        probs, decoded = track_token_prob(model, tok, text2, target, apply_ln2)
        if probs is None:
            st.error("토큰을 인코딩할 수 없습니다.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(probs))), y=probs,
                mode="lines+markers", name=f"{decoded!r} 확률"))
            fig.update_layout(
                xaxis_title="층 (0=embedding)", yaxis_title="확률",
                height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"최종 확률: **{probs[-1]:.2%}**")
            # 가장 크게 뛴 층
            jumps = [(i, probs[i] - probs[i - 1]) for i in range(1, len(probs))]
            bi, bj = max(jumps, key=lambda x: x[1])
            st.info(f"확률이 가장 크게 뛴 곳: **layer {bi}** (+{bj:.1%}). "
                    f"여기서 무슨 일이 일어났는지가 탐구 대상.")

# ── 탭3 ──
with tab3:
    st.subheader("선형화하면 뭘 잃는가")
    st.markdown("""
**질문했던 것**: 비선형 함수를 선형으로 재정렬하면, 어떤 **분기**로 설명이 이어졌는지 알 수 있나?

**logit lens가 실제로 하는 것**
- 각 층의 residual stream 벡터를, 최종 출력용 선형 사상(unembedding)에 그냥 통과시킴
- 즉 "이 층의 상태를 최종 층인 척 읽어보기"
- 이게 바로 **선형 재정렬**이고, 실제로 뭔가 읽힌다 → 위 탭들에서 확인

**그런데 여기서 나오는 것은 '흐름'이지 '분기'가 아니다**

| 기대한 것 | 실제로 있는 것 |
|---|---|
| `if A면 왼쪽, else 오른쪽` | 연속 벡터 공간의 부드러운 이동 |
| 이산적 결정 이력 | 층마다 조금씩 바뀌는 확률 분포 |
| 갈림길 | 여러 방향이 동시에 가중치를 가짐 |

**선형화가 특히 손해 보는 지점**
- `softmax` (attention): 토큰 간 경쟁이 본질. 선형으론 표현 불가
- `LayerNorm`: 스케일 의존 — 이걸 안 맞추면 초기 층이 완전히 깨짐 (탭1에서 체크박스 꺼보기)
- `GELU` 게이팅 + **superposition**: 한 뉴런이 여러 개념을 겹쳐 담음 →
  선형 분해로 뽑은 성분이 사람이 읽을 개념과 대응하지 않음

**그래서 결론**
선형 재정렬로 **기여도 궤적**은 얻는다. 하지만 **분기 추적**은 얻지 못한다 —
애초에 트랜스포머 안에 분기가 없기 때문. 없는 걸 선형화로 만들어낼 순 없다.

**대안 방향 (더 파고 싶다면)**
- `tuned lens`: 층마다 별도 선형 프로브를 **학습**시켜 오차를 줄임 (raw logit lens 개선)
- `attribution patching`: 특정 성분을 끄고 출력 변화를 봄 → 인과 기여도
- `SAE (sparse autoencoder)`: superposition을 풀어 해석 가능한 특징으로 분해

**진짜 화이트박스가 필요하면**
판단을 LLM 밖에 두면 된다. 예: 마르코프 기반 판정 엔진은
토큰별 기여도와 임계값 분기가 숫자로 완전히 열려 있다 —
근사가 아니라 계산 그 자체라서.
    """)
