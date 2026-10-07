// AI 셀프 인테리어 시각화 — 발표 슬라이드 (10분, 12장)
// 다시 만들기:
//   npm install pptxgenjs@4.0.1
//   node build.js && python3 fix_korean_wrap.py build_raw.pptx AI_셀프_인테리어_발표.pptx
// 글상자에 lang: "ko-KR"을 넣어야 PowerPoint가 한글을 단어 단위로 줄바꿈한다(없으면 "파이프라\n인").
// assets/는 이전 구현의 보존된 실험 이미지다. 정량값도 2026-09-19 당시 기록이며 수정 후 미재측정이다.
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
const history = (slide) => cap(slide, "2026-09-19 이전 구현의 실험 기록 · 09-21 수정 후 품질은 아직 재측정하지 않음", 0.5, 5.36, 9, { fontSize: 8, h: 0.14 });
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
  cap(s, "이전 구현 예시 — 0.95m 의자의 배치 (실제 크기 정답 없음)", r.x, r.y + r.h + 0.12, r.w, { color: GRAY });
  s.addNotes("AI 셀프 인테리어 시각화입니다. 방 사진과 가구 사진, 놓을 자리를 받아 그 가구가 그 자리에 놓인 모습을 만듭니다. 핵심은 이 합성을 API에 맡기지 않고 직접 구현했고, 상용 생성 AI와 비교한 과정을 기록했다는 점입니다. 사진과 정량값은 9월 19일 이전 구현의 실험이며, 9월 21일 수정 이후 품질은 아직 다시 측정하지 않았습니다.");
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
  T(s, "실제 높이(m) 입력 · 모르면 ‘수동’ 모드를 선택\n바닥 추정 실패: 세우기는 수동 진행, 러그는 명시적 오류", { x: 0.5, y: 3.72, w: 5.55, h: 0.65, fontSize: 11.5, color: MUTED });
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
  s.addNotes("방 사진은 깊이 추정과 바닥 평면 추정을 거치고, 가구 사진은 SAM으로 배경을 지웁니다. 둘을 합성 단계에서 크기와 원근, 조명, 그림자, 가림을 맞춰 합칩니다. 선택 기능으로 Stable Diffusion이 가구 주변을 다듬습니다. 전부 로컬 추론입니다. 실제 높이는 사용자가 입력하며 확인된 샘플 사진에만 저장 높이를 기본 적용합니다. 일반 가구는 자동 모드에서 높이가 비어 있으면 오류가 나므로 높이를 입력하거나 수동 모드를 선택해야 합니다. 높이 미입력으로 자동 전환하지 않습니다. 세워 놓는 가구는 바닥 추정 실패 시에만 수동 합성으로 전환하고, 바닥에 까는 러그는 바닥 추정이 필요합니다.");
}

// 4. 바닥 평면 ------------------------------------------------------------
{
  const s = pres.addSlide(); title(s, 3, "바닥 평면: 이전 구현의 외부 정답 평가"); history(s);
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
  const s = pres.addSlide(); title(s, 4, "이전 누끼 실험: 가구를 칠하면 4/10 → 8/10"); history(s);
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
  const s = pres.addSlide(); title(s, 5, "합성: 빛·그림자·가림을 근사한 실험"); history(s);
  const cards = [
    ["조명 정합", ["harm_old.jpg", "기존 방식 · 채도 6.7"], ["harm_new.jpg", "새 방식 · 채도 5.0"],
     "바닥이 아니라 방의 밝은 부분에서 조명색을 가져와, 흰 가구가 누렇게 변하던 문제를 없앴다. 기존 방식은 보정 안 한 것(5.7)보다도 나빴다.", "채도 6.7 → 5.0"],
    ["원근 반영 그림자", ["shadow_before.jpg", "상수 폭"], ["shadow_after.jpg", "바닥 평면으로 계산"],
     "이전 지평선 설정에서 상수 폭과 원근 근사를 비교했다. 실제 그림자를 측정한 정답은 아니다.", "이전 설정 7px → 3px"],
    ["깊이 기반 가림", ["occ_before.jpg", "수정 전"], ["occ_after.jpg", "수정 후"],
     "가구보다 가까운 물체(침대 기둥)가 가구를 가린다. 이전 10쌍에서 1쌍에만 작동했다. 물체 충돌이나 3D 위치를 검증한 것은 아니다.", "새 파라미터 0개"],
  ];
  cards.forEach(([h, a, b, body, stat], i) => { const x = 0.5 + i * 3.05;
    card(s, x, 1.15, 2.85, 3.95);
    T(s, h, { x: x + 0.15, y: 1.3, w: 2.55, h: 0.35, fontSize: 14, bold: true });
    [a, b].forEach(([n, t], k) => { const r = img(s, n, x + 0.12 + k * 1.33, 1.75, 1.27, 1.12, { noShadow: true });
      cap(s, t, x + 0.12 + k * 1.33, 2.95, 1.27, { fontSize: 9, align: "center", color: k ? SAGE : MUTED, bold: !!k }); });
    T(s, body, { x: x + 0.15, y: 3.35, w: 2.55, h: 1.15, fontSize: 11 });
    T(s, stat, { x: x + 0.15, y: 4.5, w: 2.55, h: 0.45, fontSize: 17, bold: true, color: ACC });
  });
  s.addNotes("이 슬라이드는 이전 구현의 실험입니다. 그림자 7→3px는 이전 지평선과 가정한 가구 깊이·초점거리의 근사이며 실제 그림자 정답과 비교한 수치가 아닙니다. 조명 채도 6.68/5.02는 DEVLOG §32의 수기 기록이며 정확한 재현 절차는 보존되지 않았습니다. 합성에서는 세 가지를 다뤘습니다. 조명은 방의 밝은 부분을 기준으로 색을 옮겨, 가구가 누렇게 변하던 문제를 없앴습니다. 그림자는 바닥 평면으로 깊이를 계산하고, 깊이맵으로 가구 앞에 있는 물체가 가구를 가리게 했습니다.");
}

// 7. 크기 ---------------------------------------------------------------
{
  const s = pres.addSlide(); title(s, 6, "크기: 문 높이를 가정한 대리 평가"); history(s);
  s.addChart(pres.charts.BAR, [
    { name: "깊이로 찾은 지평선", labels: ["<0.5", "0.5–0.75", "0.75–1.25", "1.25–2", "2–3", "3–5", ">5"], values: [0, 2, 5, 8, 15, 14, 18] },
    { name: "사진 높이 41% 가정", labels: ["<0.5", "0.5–0.75", "0.75–1.25", "1.25–2", "2–3", "3–5", ">5"], values: [1, 2, 36, 17, 4, 1, 1] }],
    { x: 0.4, y: 1.1, w: 5.0, h: 3.0, barDir: "col", barGrouping: "clustered", chartColors: [GRAY, ACC], showValue: true,
      dataLabelFontSize: 9, dataLabelColor: TEXT, dataLabelFontFace: F, valAxisHidden: true, valGridLine: { style: "none" },
      catGridLine: { style: "none" }, catAxisLabelColor: MUTED, catAxisLabelFontSize: 9.5, catAxisLabelFontFace: F,
      showLegend: true, legendPos: "t", legendFontSize: 10, legendFontFace: F, barGapWidthPct: 40 });
  cap(s, "문 2.03m · 카메라 1.4m 가정, 계산 가능 62/90개\n문 단위 분할: dev/test에 같은 방 15장 중복 · 방 단위 재검증 필요", 0.5, 4.15, 4.9, { h: 0.45 });
  T(s, [{ text: "±25% 안  ", options: { fontSize: 14, color: TEXT } }, { text: "5/62 → 36/62", options: { fontSize: 22, bold: true, color: ACC } }],
    { x: 0.5, y: 4.65, w: 4.9, h: 0.5, valign: "middle" });
  const pics = [["size_03_old.jpg", "이전: 0.95m 의자 ≈ 탁자 높이", MUTED], ["size_03_new.jpg", "수정: 의자가 탁자보다 크다", SAGE],
                ["size_09_old.jpg", "이전: 0.5m 탁자가 점", MUTED], ["size_09_new.jpg", "수정: 리클라이너 절반 높이", SAGE]];
  pics.forEach(([n, t, c], i) => { const x = 5.7 + (i % 2) * 2.0, y = 1.15 + Math.floor(i / 2) * 2.05;
    const r = img(s, n, x, y, 1.85, 1.6, { noShadow: true }); cap(s, t, x, r.y + r.h + 0.05, 1.85, { fontSize: 9, color: c, bold: c === SAGE, align: "center" }); });
  s.addNotes("이전 크기 공식의 대리 평가입니다. ADE20K가 제공한 것은 문 영역의 픽셀 정답이며 문 실제 높이 2.03m와 카메라 높이 1.4m는 가정입니다. 가로축은 문 픽셀 높이를 공식 예측으로 나눈 비율입니다. test 90개 중 계산 가능한 62개에서만 5개에서 36개로 개선됐고 나머지 28개는 분모에서 제외됐습니다. 문 단위로 나눠 dev와 test 사이 방 15장이 중복됐으므로 독립적인 새 방 성능으로 볼 수 없습니다. 현재 방 단위 분할로 다시 검증해야 하며 수정 후 결과는 아직 없습니다.");
}

// 8. CLIP 함정 ------------------------------------------------------------
{
  const s = pres.addSlide(); title(s, 7, "평가의 함정: CLIP이 벽 조각에 더 높은 점수"); history(s);
  const L = [["sam_08_local_auto.jpg", "벽 조각", "−0.0055", "−0.0062", ACC], ["sam_08_local_box.jpg", "라운지체어", "−0.0093", "+0.0148", SAGE]];
  L.forEach(([n, t, full, crop, c], i) => { const x = 0.5 + i * 2.4;
    const r = img(s, n, x, 1.15, 2.2, 1.66, { left: true });
    T(s, t, { x, y: r.y + r.h + 0.08, w: 2.2, h: 0.3, fontSize: 12, bold: true, color: c });
    T(s, [{ text: "사진 전체  " + full, options: { breakLine: true, color: i === 0 ? ACC : MUTED, bold: i === 0 } },
          { text: "배치 영역  " + crop, options: { color: i === 1 ? SAGE : MUTED, bold: i === 1 } }],
      { x, y: r.y + r.h + 0.42, w: 2.2, h: 0.6, fontSize: 11 });
  });
  const steps = [["원인", "CLIP은 사진 전체를 224px로 줄여 본다. 작은 가구의 세부 정보가 줄어든다.", ACC],
                 ["수정", "배치 영역만 잘라서 채점 → 같은 가구에서 잘된 쪽이 높은 경우 2/4 → 4/4", SAGE],
                 ["한계", "소파만 한 전등갓도 높은 점수. 이 점수로 크기의 사실감이나 제품 동일성을 입증할 수 없다", MUTED]];
  steps.forEach(([h, b, c], i) => { const y = 1.15 + i * 1.12;
    s.addShape(pres.shapes.OVAL, { x: 5.45, y, w: 0.36, h: 0.36, fill: { color: c }, line: { color: c } });
    T(s, String(i + 1), { x: 5.45, y, w: 0.36, h: 0.36, fontSize: 12, bold: true, color: WHITE, align: "center", valign: "middle" });
    T(s, h, { x: 5.95, y: y + 0.02, w: 3.5, h: 0.32, fontSize: 13, bold: true, color: c });
    T(s, b, { x: 5.95, y: y + 0.36, w: 3.55, h: 0.7, fontSize: 11.5 }); });
  T(s, "숫자만 보지 않고 결과 이미지를 직접 확인했기 때문에 찾았다", { x: 0.5, y: 4.85, w: 9, h: 0.35, fontSize: 12, italic: true, color: MUTED });
  s.addNotes("평가 지표로 CLIP을 썼는데, 의자 대신 벽 조각을 붙인 결과가 제대로 된 의자보다 점수가 높았습니다. 원인은 CLIP이 사진 전체를 작게 줄여 보기 때문이었고, 배치 영역만 잘라 채점하도록 고쳤습니다. 이 사례만으로 CLIP을 전체 품질 지표로 삼을 수 없습니다. 크기의 사실감과 제품 동일성은 검증되지 않았습니다.");
}

// 9. 로컬 vs Gemini --------------------------------------------------------
{
  const s = pres.addSlide(); title(s, 8, "로컬 vs Gemini: 이전 구현의 파일럿 3쌍"); history(s);
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
  cap(s, "픽셀 일치는 크기 변경·압축에도 달라진다. 방 구조 보존율이나 종합 품질 점수가 아니다.", 0.5, 4.95, 9.0);
  s.addNotes("9월 19일 이전 구현과 저장된 Gemini 결과 3쌍의 비교입니다. CLIP 증분 평균이며 품질이 몇 배 낫다는 뜻은 아닙니다. 원본 픽셀 완전 일치 비율은 정렬과 리사이즈·압축에 민감하므로 방 구조 보존율로 해석할 수 없습니다. 96.6%와 91.4%는 DEVLOG §32에 남은 수기 측정이며 입력별 측정 기록은 보존되지 않았습니다. 비용은 당시 API 기준, 시간은 M5 16GB의 당시 기록입니다. 수정 후 현재 품질·시간은 재측정하지 않았습니다.");
}

// 10. SD ---------------------------------------------------------------
{
  const s = pres.addSlide(); title(s, 9, "이전 다듬기 실험: 인식 가능성과 제품 동일성"); history(s);
  img(s, "sd_small.jpg", 0.5, 1.1, 9.0, 1.95, { noShadow: true });
  const cards = [["이미지 통째로 다시 그리기", "인식 가능 0/4", "28~43초", ACC_T, ACC],
                 ["가구 주변만 키워서, 강도 0.35", "인식 가능 4/4", "5~8초 · 색·무늬 변경 있음", SAGE_T, SAGE],
                 ["시험한 설정에서는", "각도는 그대로", "깊이 조건이 합성본에서 오기 때문", LIGHT, MUTED]];
  cards.forEach(([h, big, sub, bg, c], i) => { const x = 0.5 + i * 3.05;
    card(s, x, 3.3, 2.85, 1.35, bg);
    T(s, h, { x: x + 0.15, y: 3.4, w: 2.55, h: 0.3, fontSize: 11, color: MUTED });
    T(s, big, { x: x + 0.15, y: 3.72, w: 2.55, h: 0.45, fontSize: 19, bold: true, color: c });
    T(s, sub, { x: x + 0.15, y: 4.2, w: 2.55, h: 0.35, fontSize: 10.5, color: MUTED }); });
  T(s, "약하게 다듬어도 색·무늬가 바뀔 수 있다 — 실제 제품 동일성은 미검증", { x: 0.5, y: 4.85, w: 9, h: 0.4, fontSize: 14, bold: true, color: SAGE });
  s.addNotes("기획서의 Stable Diffusion과 ControlNet을 마지막 다듬기 단계로 시험했습니다. 이미지를 통째로 다시 그리면 책장이 베이지 덩어리가 되는 등 작은 가구가 전부 다른 물건이 됐고, 가구 주변만 키워서 약하게 다시 그리면 네 개 모두 알아볼 수는 있었으나 제품이 그대로 보존된 것은 아닙니다. 강도 0.35에서도 빨간 좌판이 연분홍 무늬로 바뀌었습니다. 수치는 이전 구현의 육안 판정이며 수정 후 재측정하지 않았습니다. 그래서 앱에는 기본으로 꺼진 선택 기능으로 넣었습니다.");
}

// 11. 범위와 한계 ----------------------------------------------------------
{
  const s = pres.addSlide(); title(s, 10, "범위와 한계"); history(s);
  T(s, "기획서 대비", { x: 0.5, y: 1.1, w: 4.3, h: 0.35, fontSize: 14, bold: true });
  const rows = [["구현", SAGE, SAGE_T, "깊이 추정 · SAM 누끼 · 높이 입력·수동 배치 · Before/After"],
                ["대체", "4A6FA5", "E8EEF6", "FastAPI → Gradio · CLIP 스타일 → 평가 지표 · 조명 설정 → 자동 조명 정합"],
                ["선택 기능", DARK, LIGHT, "Stable Diffusion + ControlNet 다듬기 (기본 꺼짐)"],
                ["제외", ACC, ACC_T, "스타일 변환·3D (기획서 6쪽 ‘확장 목표’) · 가격 비교 · 벽지"]];
  rows.forEach(([tag, c, bg, body], i) => { const y = 1.55 + i * 0.86;
    card(s, 0.5, y, 4.3, 0.74, bg);
    T(s, tag, { x: 0.62, y, w: 0.95, h: 0.74, fontSize: 11.5, bold: true, color: c, valign: "middle" });
    T(s, body, { x: 1.6, y, w: 3.1, h: 0.74, fontSize: 10.5, valign: "middle" }); });
  T(s, "알려진 한계", { x: 5.2, y: 1.1, w: 4.3, h: 0.35, fontSize: 14, bold: true });
  const lim = [["위에서 내려다본 사진", "크기 오차 가능 — 지평선 41%·카메라 높이 1.4m를 가정"],
               ["바닥이 적게 보이는 사진", "이전 선택 표본에서 바닥 추정 31% 실패. 현재 세우기는 수동 배치로 진행"],
               ["가는 구조", "조명의 기둥 같은 부분을 SAM이 잡지 못한다"],
               ["가구 사진의 각도", "사진에 없는 면은 만들 수 없다 — 생성 모델로도 고쳐지지 않았다"]];
  lim.forEach(([h, b], i) => { const y = 1.55 + i * 0.86;
    s.addShape(pres.shapes.OVAL, { x: 5.2, y: y + 0.08, w: 0.14, h: 0.14, fill: { color: ACC }, line: { color: ACC } });
    T(s, h, { x: 5.45, y, w: 4.05, h: 0.3, fontSize: 12, bold: true });
    T(s, b, { x: 5.45, y: y + 0.32, w: 4.05, h: 0.45, fontSize: 10.5, color: MUTED }); });
  s.addNotes("기획서 대비로 보면 핵심 합성 기능은 구현했고, 일부는 대체했으며, 스타일 변환과 3D는 기획서 스스로 확장 목표로 둔 부분이라 제외했습니다. 한계로는 위에서 내려다본 사진의 크기, 바닥이 적게 보이는 사진, 가는 구조, 가구 사진의 각도가 있습니다.");
}

// 12. 결론 ---------------------------------------------------------------
{
  const s = pres.addSlide(); s.background = { color: DARK };
  T(s, "결론", { x: 0.6, y: 0.45, w: 8, h: 0.7, fontSize: 32, bold: true, color: WHITE });
  const pts = [["기하 파이프라인은 선택한 가구 이미지를 합성한다", "위치·크기는 입력과 가정에 따른 근사이며 실제 정답은 아니다"],
               ["생성 다듬기는 색·무늬와 가구 모양을 바꿀 수 있다", "그래서 역할을 나눠, 생성은 선택적인 마감으로 쓴다"],
               ["측정의 가정과 실패를 함께 기록했다", "이전 수치와 현재 구현을 구분한다 — 수정 후 품질은 아직 미검증"]];
  pts.forEach(([h, b], i) => { const y = 1.45 + i * 1.08;
    s.addShape(pres.shapes.OVAL, { x: 0.6, y, w: 0.5, h: 0.5, fill: { color: ACC }, line: { color: ACC } });
    T(s, String(i + 1), { x: 0.6, y, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: WHITE, align: "center", valign: "middle" });
    T(s, h, { x: 1.3, y: y - 0.02, w: 8.2, h: 0.4, fontSize: 17, bold: true, color: WHITE });
    T(s, b, { x: 1.3, y: y + 0.4, w: 8.2, h: 0.35, fontSize: 13, color: "C9CDD2" }); });
  T(s, "감사합니다", { x: 0.6, y: 4.75, w: 4, h: 0.45, fontSize: 18, bold: true, color: "E88B74" });
  T(s, "github.com/yourqxc/Capstone", { x: 5.5, y: 4.8, w: 4, h: 0.4, fontSize: 11, color: GRAY, align: "right" });
  s.addNotes("기하 파이프라인은 입력한 가구 이미지를 합성하지만 위치·크기·원근·가림은 가정에 따른 근사입니다. 생성 다듬기는 제품의 색과 무늬까지 바꿀 수 있어 선택 기능으로 둡니다. 이번 수정에서는 실제 높이 입력과 수동 배치 경로를 보완했고, 발표의 과거 실험값과 현재 구현을 구분했습니다. 새 구현의 품질은 아직 다시 측정하지 않았으므로 개선 성능을 주장하지 않습니다. 감사합니다.");
}

pres.writeFile({ fileName: path.join(__dirname, "build_raw.pptx") }).then((f) => console.log("저장:", f));
