// Общее для обеих визуализаций: загрузка записанной активности мозга и расписание «обнюхиваний».
// Данные пишет export_viz.py в viz/data/.
import * as THREE from "three";

export const LETTERS = "ABCDEFGH";

const json = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(u); return r.json(); });
const bin = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(u); return r.arrayBuffer(); });

export async function load(runName) {
  const [brain, run, geo, act] = await Promise.all([
    json("data/brain.json"), json(`data/${runName}.json`), bin("data/brain.bin"), bin(`data/${runName}.bin`),
  ]);
  const N = brain.n;
  return { brain, run, N, pos: new Float32Array(geo, 0, N * 3), cls: new Uint8Array(geo, N * 12, N), A: new Uint8Array(act) };
}

/** Облако из 138 тыс. нейронов, яркость точки — частота разрядов.
 *  unit — множитель размера точек, base — прозрачность неактивного нейрона,
 *  additive — точки складываются по яркости (в маленьком облаке это даёт белое пятно — тогда false). */
export function brainCloud({ brain, run, N, pos, cls, A }, { unit = 1, base = 0.16, additive = true } = {}) {
  const K = run.k, T = run.steps, C = brain.classes.length;
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geom.setAttribute("cls", new THREE.BufferAttribute(Float32Array.from(cls), 1));
  const glow = new THREE.BufferAttribute(new Float32Array(N), 1);
  glow.setUsage(THREE.DynamicDrawUsage);
  geom.setAttribute("act", glow);
  const palette = brain.classes.map((c) => new THREE.Color(c.color));
  const material = new THREE.ShaderMaterial({
    uniforms: { palette: { value: palette }, scale: { value: 450 }, unit: { value: unit }, base: { value: base } },
    vertexShader: `
      uniform vec3 palette[${C}];
      uniform float scale, unit, base;
      attribute float cls;
      attribute float act;
      varying vec3 vColor;
      varying float vAlpha;
      void main() {
        if (cls > 200.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
        vec3 c = palette[int(cls)];
        float on = smoothstep(0.02, 0.6, act);
        vColor = mix(c * 0.28, mix(c, vec3(1.0), 0.45 * act), on);
        vAlpha = base + (1.0 - base) * on;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = (2.6 + 9.0 * act) * unit * scale / -mv.z;
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      varying vec3 vColor;
      varying float vAlpha;
      void main() {
        vec2 c = gl_PointCoord - 0.5;
        float d = dot(c, c);
        if (d > 0.25) discard;
        gl_FragColor = vec4(vColor, vAlpha * (1.0 - 4.0 * d));
      }`,
    transparent: true, depthWrite: false, blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
  });
  const count = new Float32Array(C);
  for (let i = 0; i < N; i++) if (cls[i] < C) count[cls[i]]++;
  const mean = new Float32Array(C); // средняя частота по классам нейронов на текущем кадре
  const frame = (q, o, t) => ((q * K + o) * T + t) * N;

  /** target: [вопрос, вариант, шаг] (шаг дробный, соседние кадры смешиваются) или null — затухание.
   *  Возвращает число активных нейронов. */
  function update(target) {
    const g = glow.array;
    let active = 0;
    mean.fill(0);
    if (!target) {
      for (let i = 0; i < N; i++) if ((g[i] *= 0.9) > 0.1) active++;
    } else {
      const [q, o, st] = target;
      const t0 = Math.min(Math.floor(st), T - 1), t1 = Math.min(t0 + 1, T - 1), f = st - t0;
      const a0 = frame(q, o, t0), a1 = frame(q, o, t1);
      for (let i = 0; i < N; i++) {
        const v = (A[a0 + i] * (1 - f) + A[a1 + i] * f) / 255;
        g[i] = v > g[i] ? v : g[i] * 0.88 + v * 0.12; // послесвечение
        if (v > 0.1) active++;
        if (cls[i] < C) mean[cls[i]] += v;
      }
      for (let c = 0; c < C; c++) mean[c] /= count[c] || 1;
    }
    glow.needsUpdate = true;
    return active;
  }
  return { points: new THREE.Points(geom, material), material, update, mean, clear: () => glow.array.fill(0) };
}

/** Расписание показа: муха по очереди нюхает каждый вариант, пауза, показ выбора, следующий вопрос. */
export class Player {
  constructor(run, { speed = 6, gap = 0.9, reveal = 5, q = 0 } = {}) {
    Object.assign(this, { run, speed, gap, reveal, paused: false });
    this.go(Math.min(Math.max(q, 0), run.questions.length - 1));
  }

  get item() { return this.run.questions[this.qi]; }

  go(qi) {
    Object.assign(this, { qi, k: 0, s: 0, phase: "sniff", timer: 0, changed: true });
    // шкала полос: от минимума до максимума «притяжения» после того, как пришёл запах ответа
    const after = this.item.live.flatMap((row) => row.slice(this.run.answerAt));
    this.lo = Math.min(...after);
    this.hi = Math.max(...after);
  }

  next(dq) {
    const Q = this.run.questions.length;
    this.go((this.qi + dq + Q) % Q);
  }

  tick(dt) {
    if (this.paused) return;
    const T = this.run.steps, K = this.run.k;
    if (this.phase === "sniff") {
      this.s += dt * this.speed;
      if (this.s >= T - 1) { this.s = T - 1; this.phase = "gap"; this.timer = 0; }
    } else if (this.phase === "gap") {
      if ((this.timer += dt) > this.gap) {
        if (this.k < K - 1) { this.k++; this.s = 0; this.phase = "sniff"; } else { this.phase = "reveal"; this.timer = 0; }
      }
    } else if ((this.timer += dt) > this.reveal) this.next(1);
  }

  /** Что сейчас показывать мозгу: [вопрос, вариант, шаг] или null — затухание между вариантами. */
  target() {
    if (this.phase === "sniff") return [this.qi, this.k, this.s];
    if (this.phase === "reveal") return [this.qi, this.item.chosen, this.run.steps - 1];
    return null;
  }

  /** «Притяжение» варианта i в [0, 1] на текущий момент. */
  bar(i) {
    const step = i < this.k ? this.run.steps - 1 : i === this.k ? Math.floor(this.s) : -1;
    if (step < this.run.answerAt) return 0;
    return Math.max(0, Math.min(1, (this.item.live[i][step] - this.lo) / (this.hi - this.lo || 1)));
  }

  get status() {
    const it = this.item;
    if (this.phase === "reveal") return `Муха выбрала ${LETTERS[it.chosen]} — ${it.correct ? "верно" : "неверно"}`;
    if (this.phase === "gap") return "Проветривает усики…";
    return this.s < this.run.answerAt
      ? `Нюхает вопрос · вариант ${LETTERS[this.k]} следующий`
      : `Вопрос + вариант ${LETTERS[this.k]}: тянет или нет?`;
  }
}
