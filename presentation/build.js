// AI 셀프 인테리어 시각화 — 발표 슬라이드 (10분, 13장)
// 다시 만들기:
//   npm install pptxgenjs@3
//   node build.js && python3 fix_korean_wrap.py build_raw.pptx AI_셀프_인테리어_발표.pptx
// 글상자에 lang: "ko-KR"을 넣어야 PowerPoint가 한글을 단어 단위로 줄바꿈한다(없으면 "파이프라\n인").
// assets/는 ai-interior/docs/evidence, eval/results, eval/survey/stimuli에서 잘라 온 이미지다.
const pptxgen = require("pptxgenjs");
const path = require("path");
const SIZES = require("./sizes.json");
const A = (n) => path.join(__dirname, "assets", n);

const DARK = "22262A", ACC = "C8553D", SAGE = "5E8A6A", LIGHT = "F2F3F1", TEXT = "1F2328",
      MUTED = "5F6670", GRAY = "B3B8BF", WHITE = "FFFFFF", ACC_T = "FBECE8", SAGE_T = "E9F1EB";
const F = "Malgun Gothic";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";          // 10 x 5.625 in
pres.title = "AI 셀프 인테리어 시각화";

const shadow = () => ({ type: "outer", blur: 6, offset: 2, angle: 90, color: "000000", opacity: 0.18 });
const T = (slide, text, o) => slide.addText(text, { fontFace: F, color: TEXT, margin: 0, isTextBox: true, valign: "top", lang: "ko-KR", ...o });

function img(slide, name, x, y, w, h, opts = {}) {      // 상자 안에 비율 유지, 가운데 정렬
  const [pw, ph] = SIZES[name];
  let iw = w, ih = w * ph / pw;
  if (ih > h) { ih = h; iw = h * pw / ph; }
  const ix = opts.left ? x : x + (w - iw) / 2, iy = opts.top === false ? y + (h - ih) / 2 : y;
  slide.addImage({ path: A(name), x: ix, y: iy, w: iw, h: ih, shadow: opts.noShadow ? undefined : shadow() });
  return { x: ix, y: iy, w: iw, h: ih };
}
function title(slide, num, text) {
  slide.background = { color: WHITE };
  slide.addShape(pres.shapes.OVAL, { x: 0.5, y: 0.36, w: 0.46, h: 0.46, fill: { color: ACC }, line: { color: ACC } });
  T(slide, String(num), { x: 0.5, y: 0.36, w: 0.46, h: 0.46, fontSize: 15, bold: true, color: WHITE, align: "center", valign: "middle" });
  T(slide, text, { x: 1.12, y: 0.3, w: 8.4, h: 0.58, fontSize: 24, bold: true, valign: "middle" });
}
const cap = (slide, text, x, y, w, o = {}) => T(slide, text, { x, y, w, h: 0.3, fontSize: 10.5, color: MUTED, ...o });
const card = (slide, x, y, w, h, color = LIGHT) =>
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color }, line: { color } });

// 1. 표지 ---------------------------------------------------------------
{
  const s = pres.addSlide(); s.background = { color: DARK };
  T(s, "2026 캡스톤디자인", { x: 0.6, y: 1.05, w: 4.6, h: 0.35, fontSize: 13, bold: true, color: "E88B74" });
  T(s, "AI 셀프 인테리어\n시각화", { x: 0.6, y: 1.45, w: 4.7, h: 1.5, fontSize: 36, bold: true, color: WHITE });
  T(s, "방 사진 한 장에 가구를 놓아 보는\n로컬 합성 파이프라인과\n상용 생성 AI의 비교", { x: 0.6, y: 3.05, w: 4.7, h: 1.1, fontSize: 14, color: "D5D8DC" });
  T(s, "Depth Anything · SAM · OpenCV · CLIP · Stable Diffusion", { x: 0.6, y: 4.75, w: 4.8, h: 0.3, fontSize: 10.5, color: GRAY });
  const r = img(s, "hero.jpg", 5.45, 1.05, 4.05, 3.0);
  cap(s, "로컬 파이프라인 결과 — 0.95m 의자를 지정한 자리에, 방의 원근에 맞춰", r.x, r.y + r.h + 0.12, r.w, { color: GRAY });
  s.addNotes("AI 셀프 인테리어 시각화입니다. 방 사진과 가구 사진, 놓을 자리를 받아 그 가구가 그 자리에 놓인 모습을 만듭니다. 핵심은 이 합성을 API에 맡기지 않고 직접 구현했고, 상용 생성 AI와 무엇이 다른지 측정했다는 점입니다.");
}

// 2. 출발점 ---------------------------------------------------------------
{
  const s = pres.addSlide(); title(s, 1, "출발점: 기획서와 구현이 어긋나 있었다");
  card(s, 0.5, 1.2, 3.0, 3.75);
  T(s, "0 / 8", { x: 0.5, y: 1.55, w: 3.0, h: 1.0, fontSize: 54, bold: true, color: ACC, align: "center" });
  T(s, "기획서 핵심 기술 중\n착수 시점에 구현된 수", { x: 0.7, y: 2.7, w: 2.6, h: 0.7, fontSize: 14, align: "center" });
  T(s, "PyTorch · SAM · Stable Diffusion · ControlNet · OpenCV · CLIP · Depth Estimation · FastAPI",
    { x: 0.75, y: 3.65, w: 2.5, h: 1.0, fontSize: 10, color: MUTED, align: "center" });
  card(s, 3.85, 1.2, 5.65, 1.6, ACC_T);
  T(s, "처음", { x: 4.1, y: 1.38, w: 2, h: 0.3, fontSize: 12, bold: true, color: ACC });
  T(s, "방·가구 사진을 편집 API에 그대로 넘기는 데모.\n사실상 ‘AI 중계 사이트’였다.", { x: 4.1, y: 1.72, w: 5.2, h: 0.9, fontSize: 15 });
  s.addShape(pres.shapes.DOWN_ARROW, { x: 6.45, y: 2.9, w: 0.45, h: 0.38, fill: { color: SAGE }, line: { color: SAGE } });
  card(s, 3.85, 3.35, 5.65, 1.6, SAGE_T);
  T(s, "전환", { x: 4.1, y: 3.53, w: 2, h: 0.3, fontSize: 12, bold: true, color: SAGE });
  T(s, "합성 파이프라인을 직접 구현하고,\n상용 생성 API는 비교 기준선으로만 둔다.", { x: 4.1, y: 3.87, w: 5.2, h: 0.9, fontSize: 15 });
  s.addNotes("착수 시점에 기획서가 약속한 핵심 기술 8개 중 구현된 것은 0개였습니다. 사진을 편집 API에 넘기기만 하는 구조라 중계 사이트에 가까웠습니다. 그래서 합성 과정을 직접 구현하고, 상용 API는 비교 기준선으로만 남겼습니다.");
}

// 3. 구조 ---------------------------------------------------------------
{
  const s = pres.addSlide(); title(s, 2, "구조: 모두 이 컴퓨터에서 도는 합성 파이프라인");
  const box = (x, y, w, h, head, sub, fill = LIGHT, color = TEXT, dash) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: fill },
      line: { color: dash ? SAGE : fill, width: dash ? 1.25 : 0.75, dashType: dash ? "dash" : "solid" } });
    T(s, [{ text: head, options: { bold: true, fontSize: 12.5, color, breakLine: !!sub } },
          ...(sub ? [{ text: sub, options: { fontSize: 10, color: color === WHITE ? "E6E6E6" : MUTED } }] : [])],
      { x: x + 0.08, y, w: w - 0.16, h, align: "center", valign: "middle" });
  };
  const arrow = (x1, y1, x2, y2) => s.addShape(pres.shapes.LINE, { x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color: GRAY, width: 1.5, endArrowType: "triangle" } });
  const r1 = 1.3, r2 = 2.55, bh = 0.85;
  box(0.5, r1, 1.2, bh, "방 사진", "");
  box(0.5, r2, 1.2, bh, "가구 사진", "");
  box(2.05, r1, 1.85, bh, "깊이 추정", "Depth Anything V2");
  box(4.25, r1, 1.85, bh, "바닥 평면", "역깊이 RANSAC");
  box(2.05, r2, 1.85, bh, "누끼", "SAM ViT-B · 박스 프롬프트");
  box(6.45, r1, 1.75, r2 + bh - r1, "합성", "크기·원근 배치\n조명 정합\n접지 그림자\n깊이 기반 가림", DARK, WHITE);
  box(8.55, r1, 0.95, r2 + bh - r1, "결과", "", ACC, WHITE);
  box(6.45, 3.95, 1.75, 0.72, "선택: AI 다듬기", "SD 1.5 + ControlNet", WHITE, TEXT, true);
  arrow(1.7, r1 + bh / 2, 2.05, r1 + bh / 2); arrow(3.9, r1 + bh / 2, 4.25, r1 + bh / 2); arrow(6.1, r1 + bh / 2, 6.45, r1 + bh / 2);
  arrow(1.7, r2 + bh / 2, 2.05, r2 + bh / 2); arrow(3.9, r2 + bh / 2, 6.45, r2 + bh / 2); arrow(8.2, (r1 + r2 + bh) / 2, 8.55, (r1 + r2 + bh) / 2);
  s.addShape(pres.shapes.LINE, { x: 7.325, y: r2 + bh, w: 0, h: 3.95 - (r2 + bh), line: { color: SAGE, width: 1.25, dashType: "dash", endArrowType: "triangle" } });
  s.addShape(pres.shapes.LINE, { x: 8.2, y: 4.31, w: 0.825, h: 0, line: { color: SAGE, width: 1.25, dashType: "dash" } });
  s.addShape(pres.shapes.LINE, { x: 9.025, y: r2 + bh, w: 0, h: 4.31 - (r2 + bh), line: { color: SAGE, width: 1.25, dashType: "dash", beginArrowType: "triangle" } });
  const chips = [["PyTorch (MPS)", 1.25], ["Depth Anything V2", 1.55], ["SAM", 0.6], ["OpenCV", 0.85], ["CLIP (평가)", 1.05], ["Stable Diffusion + ControlNet", 2.3]];
  let cx = 0.5;
  chips.forEach(([c, w]) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: 4.85, w, h: 0.34, rectRadius: 0.08, fill: { color: SAGE_T }, line: { color: SAGE_T } });
    T(s, c, { x: cx, y: 4.85, w, h: 0.34, fontSize: 10, color: SAGE, bold: true, align: "center", valign: "middle" }); cx += w + 0.12; });
  s.addNotes("방 사진은 깊이 추정과 바닥 평면 추정을 거치고, 가구 사진은 SAM으로 배경을 지웁니다. 둘을 합성 단계에서 크기와 원근, 조명, 그림자, 가림을 맞춰 합칩니다. 선택 기능으로 Stable Diffusion이 가구 주변을 다듬습니다. 전부 M5 16GB 맥에서 로컬로 돕니다.");
}

// 4. 바닥 평면 ------------------------------------------------------------
{
  const s = pres.addSlide(); title(s, 3, "바닥 평면: 외부 정답으로 채점했다");
  s.addChart(pres.charts.BAR, [{ name: "IoU", labels: ["초기", "제약 2종", "성분 처리", "허용오차 조정", "새 방 10장"],
    values: [0.492, 0.646, 0.679, 0.864, 0.784] }], {
    x: 0.4, y: 1.15, w: 4.8, h: 3.35, barDir: "col", chartColors: [ACC], showValue: true, dataLabelFormatCode: "0.00",
    dataLabelColor: TEXT, dataLabelFontSize: 11, dataLabelFontFace: F, valAxisMinVal: 0, valAxisMaxVal: 1, valAxisHidden: true,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, catAxisLabelColor: MUTED, catAxisLabelFontSize: 10,
    catAxisLabelFontFace: F, showLegend: false, barGapWidthPct: 60 });
  cap(s, "ADE20K 정답 바닥과의 IoU (1이면 완벽). 마지막 막대는 튜닝에 한 번도 쓰지 않은 방", 0.5, 4.6, 4.7, { h: 0.5 });
  const r = img(s, "floor.jpg", 5.4, 1.2, 4.1, 1.9);
  cap(s, "초록 = 추정한 바닥, 선 = 계산한 지평선", r.x, r.y + r.h + 0.08, r.w);
  T(s, [{ text: "원리  ", options: { bold: true, color: SAGE } }, { text: "바닥 같은 평면에서는 역깊이가 화면 좌표의 1차식이다. 깊이맵에서 그 식에 맞는 픽셀을 RANSAC으로 찾는다", options: {} }],
    { x: 5.4, y: 3.45, w: 4.1, h: 0.75, fontSize: 12 });
  T(s, [{ text: "교훈  ", options: { bold: true, color: ACC } }, { text: "0.68 → 0.86은 새 규칙이 아니라 이미 있던 허용오차를 훑어서 얻었다.", options: {} }],
    { x: 5.4, y: 4.3, w: 4.1, h: 0.75, fontSize: 12 });
  s.addNotes("바닥 추정은 ADE20K 정답으로 채점했습니다. 0.49에서 시작해 0.86까지 올렸는데, 가장 큰 개선은 새 규칙이 아니라 기존 허용오차를 훑어서 나왔습니다. 0.86은 튜닝에 쓴 방의 값이라, 한 번도 쓰지 않은 방 10장으로 다시 재서 0.78을 얻었습니다.");
}

// 5. 누끼 ---------------------------------------------------------------
{
  const s = pres.addSlide(); title(s, 4, "누끼: 가구를 칠하면 4/10 → 8/10");
  const cells = [["sam_03_local_auto.jpg", "칠하지 않음: 좌판만", ACC], ["sam_03_local_box.jpg", "칠함: 의자 전체", SAGE],
                 ["sam_08_local_auto.jpg", "칠하지 않음: 뒤쪽 벽 조각", ACC], ["sam_08_local_box.jpg", "칠함: 라운지체어", SAGE]];
  cells.forEach(([n, t, c], i) => { const x = 0.5 + (i % 2) * 2.6, y = 1.15 + Math.floor(i / 2) * 2.1;
    const r = img(s, n, x, y, 2.4, 1.72, { left: true }); cap(s, t, r.x, r.y + r.h + 0.06, 2.4, { color: c, bold: true, fontSize: 11 }); });
  T(s, "4 → 8", { x: 5.9, y: 1.2, w: 3.6, h: 0.9, fontSize: 48, bold: true, color: ACC });
  T(s, "누끼가 제대로 나온 가구 수 (10개 중, 육안)", { x: 5.9, y: 2.1, w: 3.6, h: 0.35, fontSize: 12, color: MUTED });
  T(s, [
    { text: "SAM에 가구를 감싸는 박스를 주면 정확해진다 — 박스 7/10 > 중앙점 5/10 > 전체 2/10", options: { bullet: { indent: 12 }, breakLine: true } },
    { text: "앱의 샘플은 미리 정해 둔 박스를 쓴다 (칠하지 않아도)", options: { bullet: { indent: 12 }, breakLine: true } },
    { text: "박스는 누끼 결과를 보기 전에 정했다 — 결과를 보고 고르면 평가셋에 맞춘 선택", options: { bullet: { indent: 12 } } },
  ], { x: 5.9, y: 2.65, w: 3.6, h: 2.3, fontSize: 12, paraSpaceAfter: 8 });
  s.addNotes("가구 사진에서 가구만 떼어 내는 누끼는 SAM을 씁니다. 가구를 칠하지 않으면 의자가 좌판만 남거나 뒤쪽 벽 조각을 잡았고, 가구를 감싸게 칠하면 10개 중 8개가 제대로 나왔습니다. 박스는 결과를 보기 전에 정했습니다.");
}

// 6. 합성 ---------------------------------------------------------------
{
  const s = pres.addSlide(); title(s, 5, "합성: 방의 빛과 깊이에 맞추기");
  const cards = [
    ["조명 정합", ["harm_old.jpg", "기존 방식 · 채도 6.7"], ["harm_new.jpg", "새 방식 · 채도 5.0"],
     "바닥이 아니라 방의 밝은 부분에서 조명색을 가져와, 흰 가구가 누렇게 변하던 문제를 없앴다. 기존 방식은 보정 안 한 것(5.7)보다도 나빴다.", "채도 6.7 → 5.0"],
    ["원근 반영 그림자", ["shadow_before.jpg", "상수 폭"], ["shadow_after.jpg", "바닥 평면으로 계산"],
     "먼 곳에 놓을수록 그림자가 실제보다 두 배 깊게 깔리던 것을 바닥 평면 기준으로 바로잡았다.", "먼 배치 7px → 3px"],
    ["깊이 기반 가림", ["occ_before.jpg", "수정 전"], ["occ_after.jpg", "수정 후"],
     "가구보다 가까운 물체(침대 기둥)가 가구를 가린다. 새 파라미터 없이, 10쌍 중 필요한 1쌍에서만 작동했다.", "새 파라미터 0개"],
  ];
  cards.forEach(([h, a, b, body, stat], i) => { const x = 0.5 + i * 3.05;
    card(s, x, 1.15, 2.85, 3.95);
    T(s, h, { x: x + 0.15, y: 1.3, w: 2.55, h: 0.35, fontSize: 14, bold: true });
    [a, b].forEach(([n, t], k) => { const r = img(s, n, x + 0.12 + k * 1.33, 1.75, 1.27, 1.12, { noShadow: true });
      cap(s, t, x + 0.12 + k * 1.33, 2.95, 1.27, { fontSize: 9, align: "center", color: k ? SAGE : MUTED, bold: !!k }); });
    T(s, body, { x: x + 0.15, y: 3.35, w: 2.55, h: 1.15, fontSize: 11 });
    T(s, stat, { x: x + 0.15, y: 4.5, w: 2.55, h: 0.45, fontSize: 17, bold: true, color: ACC });
  });
  s.addNotes("합성에서는 세 가지를 맞췄습니다. 조명은 방의 밝은 부분을 기준으로 색을 옮겨, 가구가 누렇게 변하던 문제를 없앴습니다. 그림자는 바닥 평면으로 깊이를 계산하고, 깊이맵으로 가구 앞에 있는 물체가 가구를 가리게 했습니다.");
}

// 7. 크기 ---------------------------------------------------------------
{
  const s = pres.addSlide(); title(s, 6, "크기: 정답으로 재 보니 3배 작았다");
  s.addChart(pres.charts.BAR, [
    { name: "깊이로 찾은 지평선", labels: ["<0.5", "0.5–0.75", "0.75–1.25", "1.25–2", "2–3", "3–5", ">5"], values: [0, 2, 5, 8, 15, 14, 18] },
    { name: "사진 높이 41% 가정", labels: ["<0.5", "0.5–0.75", "0.75–1.25", "1.25–2", "2–3", "3–5", ">5"], values: [1, 2, 36, 17, 4, 1, 1] }],
    { x: 0.4, y: 1.1, w: 5.0, h: 3.0, barDir: "col", barGrouping: "clustered", chartColors: [GRAY, ACC], showValue: true,
      dataLabelFontSize: 9, dataLabelColor: TEXT, dataLabelFontFace: F, valAxisHidden: true, valGridLine: { style: "none" },
      catGridLine: { style: "none" }, catAxisLabelColor: MUTED, catAxisLabelFontSize: 9.5, catAxisLabelFontFace: F,
      showLegend: true, legendPos: "t", legendFontSize: 10, legendFontFace: F, barGapWidthPct: 40 });
  cap(s, "ADE20K 정답의 문(높이 2.03m) 62개, 방법을 고를 때 쓰지 않은 절반\n가로축: 실측 높이 ÷ 공식 높이 (1이면 정확)", 0.5, 4.15, 4.9, { h: 0.45 });
  T(s, [{ text: "±25% 안  ", options: { fontSize: 14, color: TEXT } }, { text: "5/62 → 36/62", options: { fontSize: 22, bold: true, color: ACC } }],
    { x: 0.5, y: 4.65, w: 4.9, h: 0.5, valign: "middle" });
  const pics = [["size_03_old.jpg", "이전: 0.95m 의자 ≈ 탁자 높이", MUTED], ["size_03_new.jpg", "수정: 의자가 탁자보다 크다", SAGE],
                ["size_09_old.jpg", "이전: 0.5m 탁자가 점", MUTED], ["size_09_new.jpg", "수정: 리클라이너 절반 높이", SAGE]];
  pics.forEach(([n, t, c], i) => { const x = 5.7 + (i % 2) * 2.0, y = 1.15 + Math.floor(i / 2) * 2.05;
    const r = img(s, n, x, y, 1.85, 1.6, { noShadow: true }); cap(s, t, x, r.y + r.h + 0.05, 1.85, { fontSize: 9, color: c, bold: c === SAGE, align: "center" }); });
  s.addNotes("크기는 지평선 공식으로 계산합니다. 그런데 사진 속 문을 자로 삼아 180개를 재 보니, 깊이맵으로 찾은 지평선으로는 ±25% 안이 62개 중 5개뿐이었고 대부분 3배쯤 작았습니다. 사람들이 방을 눈높이에서 수평으로 찍는다는 관행, 즉 지평선을 사진 높이 41%로 두는 단순한 가정이 36개로 훨씬 정확했습니다. 이 값은 절반의 데이터로 정하고 나머지 절반으로 확인했습니다.");
}

// 8. CLIP 함정 ------------------------------------------------------------
{
  const s = pres.addSlide(); title(s, 7, "평가의 함정: CLIP이 벽 조각에 더 높은 점수를 줬다");
  const L = [["sam_08_local_auto.jpg", "벽 조각", "−0.0055", "−0.0062", ACC], ["sam_08_local_box.jpg", "라운지체어", "−0.0093", "+0.0148", SAGE]];
  L.forEach(([n, t, full, crop, c], i) => { const x = 0.5 + i * 2.4;
    const r = img(s, n, x, 1.15, 2.2, 1.66, { left: true });
    T(s, t, { x, y: r.y + r.h + 0.08, w: 2.2, h: 0.3, fontSize: 12, bold: true, color: c });
    T(s, [{ text: "사진 전체  " + full, options: { breakLine: true, color: i === 0 ? ACC : MUTED, bold: i === 0 } },
          { text: "배치 영역  " + crop, options: { color: i === 1 ? SAGE : MUTED, bold: i === 1 } }],
      { x, y: r.y + r.h + 0.42, w: 2.2, h: 0.6, fontSize: 11 });
  });
  const steps = [["원인", "CLIP은 사진 전체를 224px로 줄여 본다. 화면의 3%인 가구는 몇 픽셀이 된다.", ACC],
                 ["수정", "배치 영역만 잘라서 채점 → 같은 가구에서 잘된 쪽이 높은 경우 2/4 → 4/4", SAGE],
                 ["한계", "소파만 한 전등갓도 높은 점수. CLIP은 ‘크기가 말이 되는가’를 못 본다 → 사람 평가로", MUTED]];
  steps.forEach(([h, b, c], i) => { const y = 1.15 + i * 1.12;
    s.addShape(pres.shapes.OVAL, { x: 5.45, y, w: 0.36, h: 0.36, fill: { color: c }, line: { color: c } });
    T(s, String(i + 1), { x: 5.45, y, w: 0.36, h: 0.36, fontSize: 12, bold: true, color: WHITE, align: "center", valign: "middle" });
    T(s, h, { x: 5.95, y: y + 0.02, w: 3.5, h: 0.32, fontSize: 13, bold: true, color: c });
    T(s, b, { x: 5.95, y: y + 0.36, w: 3.55, h: 0.7, fontSize: 11.5 }); });
  T(s, "숫자만 보지 않고 결과 이미지를 직접 확인했기 때문에 찾았다", { x: 0.5, y: 4.85, w: 9, h: 0.35, fontSize: 12, italic: true, color: MUTED });
  s.addNotes("평가 지표로 CLIP을 썼는데, 의자 대신 벽 조각을 붙인 결과가 제대로 된 의자보다 점수가 높았습니다. 원인은 CLIP이 사진 전체를 작게 줄여 보기 때문이었고, 배치 영역만 잘라 채점하도록 고쳤습니다. 하지만 크기의 사실감은 여전히 못 보므로 사람 평가를 준비했습니다.");
}

// 9. 로컬 vs Gemini --------------------------------------------------------
{
  const s = pres.addSlide(); title(s, 8, "로컬 vs Gemini: 무엇을 얻고 무엇을 잃나");
  const cols = [["cmp_A03l.jpg", "로컬"], ["cmp_A03lr.jpg", "로컬 + AI 다듬기"], ["cmp_A03g.jpg", "Gemini (유료 생성)"]];
  cols.forEach(([n, t], i) => { const x = 0.5 + i * 3.075; const r = img(s, n, x, 1.1, 2.85, 2.2);
    T(s, t, { x, y: r.y + r.h + 0.06, w: 2.85, h: 0.3, fontSize: 12, bold: true, align: "center", color: i === 2 ? ACC : TEXT }); });
  const hdr = (t) => ({ text: t, options: { bold: true, color: MUTED, fontSize: 10.5 } });
  const c = (t, o = {}) => ({ text: t, options: { align: "center", fontSize: 11.5, ...o } });
  s.addTable([
    [hdr("배치 영역 CLIP (3쌍 평균)"), c("+0.051"), c("+0.056"), c("+0.083", { bold: true, color: ACC })],
    [hdr("원본 사진 픽셀이 그대로인 비율"), c("약 97%", { bold: true, color: SAGE }), c("약 91%", { bold: true, color: SAGE }), c("1% 미만")],
    [hdr("비용 · 시간"), c("0원 · 몇 초"), c("0원 · +5~13초"), c("장당 $0.067 · 10~30초")],
  ], { x: 0.5, y: 3.75, w: 9.0, colW: [2.55, 2.15, 2.15, 2.15], rowH: 0.36, fontFace: F, color: TEXT,
       border: { type: "solid", pt: 0.75, color: "DADDE1" }, fill: { color: WHITE }, valign: "middle" });
  cap(s, "Gemini는 가구를 더 크게 그리고 방 전체를 다시 그린다. CLIP은 크기와 원본 유지를 보지 않는다.", 0.5, 4.95, 9.0);
  s.addNotes("같은 방과 의자를 세 방식으로 합성했습니다. CLIP 점수는 Gemini가 높지만, Gemini는 방 전체를 다시 그려서 원본 픽셀이 1%도 남지 않고 비용과 시간이 듭니다. 로컬은 원본 픽셀의 97%를 그대로 두고 가구만 넣으며 무료입니다. 다듬기를 켜면 가구 주변을 다시 그려 91%가 됩니다. 크기 공식을 고친 뒤 격차는 약 1.6배로 줄었습니다.");
}

// 10. SD ---------------------------------------------------------------
{
  const s = pres.addSlide(); title(s, 9, "생성 모델을 붙여 보니: 자연스러움 ↔ 가구 보존");
  img(s, "sd_small.jpg", 0.5, 1.1, 9.0, 1.95, { noShadow: true });
  const cards = [["이미지 통째로 다시 그리기", "가구 보존 0/4", "28~43초", ACC_T, ACC],
                 ["가구 주변만 키워서, 강도 0.35", "가구 보존 4/4", "5~8초", SAGE_T, SAGE],
                 ["어느 설정에서도", "각도는 그대로", "깊이 조건이 합성본에서 오기 때문", LIGHT, MUTED]];
  cards.forEach(([h, big, sub, bg, c], i) => { const x = 0.5 + i * 3.05;
    card(s, x, 3.3, 2.85, 1.35, bg);
    T(s, h, { x: x + 0.15, y: 3.4, w: 2.55, h: 0.3, fontSize: 11, color: MUTED });
    T(s, big, { x: x + 0.15, y: 3.72, w: 2.55, h: 0.45, fontSize: 19, bold: true, color: c });
    T(s, sub, { x: x + 0.15, y: 4.2, w: 2.55, h: 0.35, fontSize: 10.5, color: MUTED }); });
  T(s, "기하가 ‘무엇을 · 어디에 · 어떤 크기로’를 정하고, 생성 모델은 마감만 한다", { x: 0.5, y: 4.85, w: 9, h: 0.4, fontSize: 14, bold: true, color: SAGE });
  s.addNotes("기획서의 Stable Diffusion과 ControlNet을 마지막 다듬기 단계로 시험했습니다. 이미지를 통째로 다시 그리면 책장이 베이지 덩어리가 되는 등 작은 가구가 전부 다른 물건이 됐고, 가구 주변만 키워서 약하게 다시 그리면 가구가 유지됐습니다. 그래서 앱에는 기본으로 꺼진 선택 기능으로 넣었습니다.");
}

// 11. 설문 ---------------------------------------------------------------
{
  const s = pres.addSlide(); title(s, 10, "사람 평가: CLIP이 못 재는 것을 사람에게");
  card(s, 0.5, 1.1, 4.3, 3.95);
  const r1 = img(s, "survey_room.jpg", 0.65, 1.25, 2.3, 1.55, { left: true, noShadow: true });
  cap(s, "방 (빨간 표시 = 놓을 자리)", 0.65, r1.y + r1.h + 0.03, 2.3, { fontSize: 9 });
  const r2 = img(s, "survey_item.jpg", 3.1, 1.25, 1.5, 1.55, { noShadow: true });
  cap(s, "가구 사진", 3.1, r2.y + r2.h + 0.03, 1.5, { fontSize: 9, align: "center" });
  const r3 = img(s, "survey_result.jpg", 0.65, 3.15, 4.0, 1.6, { noShadow: true });
  cap(s, "합성 결과 — 어느 방식인지 표시하지 않는다", 0.65, r3.y + r3.h + 0.03, 4.0, { fontSize: 9, align: "center" });
  const blocks = [["19문항", ["방식 비교 9 — 로컬 / 로컬 + 다듬기 / Gemini", "다듬기 효과 10 — 다듬기 없음 / 있음"]],
                  ["문항마다 1~5점", ["① 표시한 자리에 놓였나", "② 그 방의 실제 가구처럼 자연스러운가", "③ 원래 가구와 같은 제품인가"]],
                  ["공정하게", ["방식을 숨기고, 순서를 사람마다 섞는다", "계정 없이 링크로 답하고, 응답은 이 컴퓨터에만"]]];
  let by = 1.15;
  blocks.forEach(([h, lines]) => {
    T(s, h, { x: 5.1, y: by, w: 4.4, h: 0.32, fontSize: 12.5, bold: true, color: ACC });
    T(s, lines.join("\n"), { x: 5.1, y: by + 0.33, w: 4.4, h: 0.26 * lines.length + 0.05, fontSize: 11.5 });
    by += 0.33 + 0.26 * lines.length + 0.22; });
  card(s, 5.1, 4.35, 4.4, 0.62, ACC_T);
  T(s, "응답 수집 중 — 결과는 발표 전에 이 자리에 채운다", { x: 5.25, y: 4.35, w: 4.1, h: 0.62, fontSize: 12, bold: true, color: ACC, valign: "middle" });
  s.addNotes("자동 지표로는 사실감과 가구 보존을 잴 수 없어서 사람 평가를 설계했습니다. 19문항에 위치, 자연스러움, 가구 보존을 1에서 5점으로 받고, 어느 방식인지 숨기고 순서를 섞습니다. 현재 응답을 모으는 중입니다.");
}

// 12. 범위와 한계 ----------------------------------------------------------
{
  const s = pres.addSlide(); title(s, 11, "범위와 한계");
  T(s, "기획서 대비", { x: 0.5, y: 1.1, w: 4.3, h: 0.35, fontSize: 14, bold: true });
  const rows = [["구현", SAGE, SAGE_T, "깊이 추정 · SAM 누끼 · 원근·크기 자동 보정 · Before/After"],
                ["대체", "4A6FA5", "E8EEF6", "FastAPI → Gradio · CLIP 스타일 → 평가 지표 · 조명 설정 → 자동 조명 정합"],
                ["선택 기능", DARK, LIGHT, "Stable Diffusion + ControlNet 다듬기 (기본 꺼짐)"],
                ["제외", ACC, ACC_T, "스타일 변환·3D (기획서 6쪽 ‘확장 목표’) · 가격 비교 · 벽지"]];
  rows.forEach(([tag, c, bg, body], i) => { const y = 1.55 + i * 0.86;
    card(s, 0.5, y, 4.3, 0.74, bg);
    T(s, tag, { x: 0.62, y, w: 0.95, h: 0.74, fontSize: 11.5, bold: true, color: c, valign: "middle" });
    T(s, body, { x: 1.6, y, w: 3.1, h: 0.74, fontSize: 10.5, valign: "middle" }); });
  T(s, "알려진 한계", { x: 5.2, y: 1.1, w: 4.3, h: 0.35, fontSize: 14, bold: true });
  const lim = [["위에서 내려다본 사진", "크기가 틀린다 — 지평선을 사진 높이 41%로 가정하기 때문"],
               ["바닥이 적게 보이는 사진", "바닥 평면 실패: 일반 실내 사진 31%, 바닥이 10% 이상 보이면 17%"],
               ["가는 구조", "조명의 기둥 같은 부분을 SAM이 잡지 못한다"],
               ["가구 사진의 각도", "사진에 없는 면은 만들 수 없다 — 생성 모델로도 고쳐지지 않았다"]];
  lim.forEach(([h, b], i) => { const y = 1.55 + i * 0.86;
    s.addShape(pres.shapes.OVAL, { x: 5.2, y: y + 0.08, w: 0.14, h: 0.14, fill: { color: ACC }, line: { color: ACC } });
    T(s, h, { x: 5.45, y, w: 4.05, h: 0.3, fontSize: 12, bold: true });
    T(s, b, { x: 5.45, y: y + 0.32, w: 4.05, h: 0.45, fontSize: 10.5, color: MUTED }); });
  s.addNotes("기획서 대비로 보면 핵심 합성 기능은 구현했고, 일부는 대체했으며, 스타일 변환과 3D는 기획서 스스로 확장 목표로 둔 부분이라 제외했습니다. 한계로는 위에서 내려다본 사진의 크기, 바닥이 적게 보이는 사진, 가는 구조, 가구 사진의 각도가 있습니다.");
}

// 13. 결론 ---------------------------------------------------------------
{
  const s = pres.addSlide(); s.background = { color: DARK };
  T(s, "결론", { x: 0.6, y: 0.45, w: 8, h: 0.7, fontSize: 32, bold: true, color: WHITE });
  const pts = [["기하 파이프라인은 가구를 그대로 지키며", "위치 · 크기 · 원근 · 가림을 정한다"],
               ["생성 모델은 자연스러움을 더하지만 가구를 바꾼다", "그래서 역할을 나눠, 생성은 선택적인 마감으로 쓴다"],
               ["숫자를 믿기 전에 정답으로 검증했다", "크기 3배 오차, CLIP의 함정, 틀린 가설 — 모두 측정으로 찾았다"]];
  pts.forEach(([h, b], i) => { const y = 1.45 + i * 1.08;
    s.addShape(pres.shapes.OVAL, { x: 0.6, y, w: 0.5, h: 0.5, fill: { color: ACC }, line: { color: ACC } });
    T(s, String(i + 1), { x: 0.6, y, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: WHITE, align: "center", valign: "middle" });
    T(s, h, { x: 1.3, y: y - 0.02, w: 8.2, h: 0.4, fontSize: 17, bold: true, color: WHITE });
    T(s, b, { x: 1.3, y: y + 0.4, w: 8.2, h: 0.35, fontSize: 13, color: "C9CDD2" }); });
  T(s, "감사합니다", { x: 0.6, y: 4.75, w: 4, h: 0.45, fontSize: 18, bold: true, color: "E88B74" });
  T(s, "github.com/yourqxc/Capstone", { x: 5.5, y: 4.8, w: 4, h: 0.4, fontSize: 11, color: GRAY, align: "right" });
  s.addNotes("정리하면, 기하 파이프라인은 가구를 그대로 지키면서 위치와 크기, 가림을 정확히 정하고, 생성 모델은 자연스러움을 더하는 대신 가구를 바꿉니다. 그래서 둘의 역할을 나눴습니다. 그리고 이 과정에서 크기 오차나 평가 지표의 함정처럼 중요한 문제는 모두 정답과 대조한 측정으로 찾았습니다. 감사합니다.");
}

pres.writeFile({ fileName: path.join(__dirname, "build_raw.pptx") }).then((f) => console.log("저장:", f));
