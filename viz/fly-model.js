// Низкополигональная муха: грудь, брюшко, голова с фасеточными глазами, усики, крылья, шесть лапок.
// Модель вынесена из desk.html (муха за столом), чтобы сцена и анимация мухи не жили в одном файле
// и не разъезжались при правках.
import * as THREE from "three";

const V = THREE.Vector3;

// детерминированный генератор: щетинки на груди одинаковые при каждой загрузке
let seed = 7;
const rng = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
const rnd = (a, b) => a + rng() * (b - a);

export const std = (color, extra = {}) =>
  new THREE.MeshStandardMaterial({ color, flatShading: true, roughness: 0.85, ...extra });
export const neon = (color) => new THREE.MeshBasicMaterial({ color });

/** Цилиндр от точки a до точки b — сегмент лапки или усика. */
export function limb(a, b, r, mat) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.8, r, a.distanceTo(b), 5), mat);
  m.position.copy(a).lerp(b, 0.5);
  m.quaternion.setFromUnitVectors(new V(0, 1, 0), b.clone().sub(a).normalize());
  return m;
}

/** Муха смотрит вдоль +X, стоит на y = 0, длина около 7 единиц — масштабируйте под свою сцену. */
export function buildFly() {
  const fly = new THREE.Group();
  const shell = std(0xa9bcb8), shellDark = std(0x74858c), legs = std(0x39424e);
  const eyeMat = std(0xc3162c, { emissive: 0x3a0008 });
  const ico = (r, mat, parent, x, y, z, sx = 1, sy = 1, sz = 1) => {
    const m = new THREE.Mesh(new THREE.IcosahedronGeometry(r, 1), mat);
    m.position.set(x, y, z);
    m.scale.set(sx, sy, sz);
    parent.add(m);
    return m;
  };
  ico(1.5, shell, fly, 0, 2.7, 0, 1.25, 1, 1);                                // грудь
  ico(1.4, shellDark, fly, -2.7, 2.35, 0, 1.8, 0.95, 1.05).rotation.z = 0.12; // брюшко
  const head = new THREE.Group();
  head.position.set(1.95, 2.95, 0);
  fly.add(head);
  ico(0.95, shell, head, 0.1, 0, 0, 0.8, 1, 1.15);
  for (const s of [-1, 1]) ico(0.78, eyeMat, head, 0.25, 0.12, s * 0.62, 0.85, 1.15, 0.7); // фасеточные глаза
  const antennae = [-1, 1].map((s) => {
    const a = new THREE.Group();
    a.position.set(0.8, 0.4, s * 0.2);
    head.add(a);
    a.add(limb(new V(), new V(0.4, 0.3, s * 0.1), 0.07, legs),
          limb(new V(0.4, 0.3, s * 0.1), new V(0.75, 0.85, s * 0.3), 0.035, legs));
    return a;
  });

  // крылья: flap машет вокруг оси тела, spread разводит назад-в стороны, левое — зеркало правого
  const wingShape = new THREE.Shape([[0, 0], [4.9, -0.5], [5.8, 0.3], [5.3, 1.4], [2.3, 1.5], [0.3, 0.5]]
    .map(([x, y]) => new THREE.Vector2(x, y)));
  const wingMat = new THREE.MeshStandardMaterial({
    color: 0xcfe4f0, transparent: true, opacity: 0.5, side: THREE.DoubleSide,
    flatShading: true, roughness: 0.4, depthWrite: false,
  });
  const wingLines = new THREE.LineBasicMaterial({ color: 0xe8f4ff, transparent: true, opacity: 0.75 });
  const veins = new THREE.BufferGeometry().setFromPoints(
    [[0.3, 0.3, 5.3, 0.1], [0.3, 0.5, 4.9, 1.2], [1.8, 0.2, 3.4, 1.4], [3.2, 0.1, 4.4, 1.35]]
      .flatMap(([x1, y1, x2, y2]) => [new V(x1, y1, 0), new V(x2, y2, 0)]));
  const wings = [-1, 1].map((s) => {
    const flap = new THREE.Group();
    flap.position.set(-0.3, 3.85, s * 0.45);
    flap.scale.z = s;
    fly.add(flap);
    const spread = new THREE.Group();
    spread.rotation.y = Math.PI + 0.32;
    flap.add(spread);
    const w = new THREE.Mesh(new THREE.ShapeGeometry(wingShape), wingMat);
    w.rotation.x = -Math.PI / 2 + 0.12;
    w.scale.setScalar(0.85);
    w.add(new THREE.LineSegments(new THREE.EdgesGeometry(w.geometry), wingLines),
          new THREE.LineSegments(veins, wingLines));
    spread.add(w);
    return flap;
  });

  const legGroups = [];
  for (const s of [-1, 1]) {
    [1.0, 0.1, -0.8].forEach((x, i) => {
      const hip = new V(x, 2.0, s * 0.8);
      const knee = new V(x + (1 - i) * 1.2 + 0.2, 3.0, s * 2.4).sub(hip);
      const foot = new V(x + (1 - i) * 2.1, 0.05, s * 3.1).sub(hip);
      const leg = new THREE.Group();
      leg.position.copy(hip);
      leg.add(limb(new V(), knee, 0.13, legs), limb(knee, foot, 0.1, legs));
      fly.add(leg);
      legGroups.push(leg);
    });
  }

  const bristles = [];
  for (let i = 0; i < 16; i++) {
    const x = rnd(-1.3, 1.3), z = rnd(-0.9, 0.9);
    const y = 2.7 + Math.sqrt(Math.max(0, 1 - (x / 1.9) ** 2 - (z / 1.5) ** 2)) * 1.45;
    bristles.push(new V(x, y, z), new V(x - 0.35, y + 0.5, z * 1.25));
  }
  fly.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(bristles),
    new THREE.LineBasicMaterial({ color: 0x0b0b12 })));
  return { fly, head, antennae, wings, legs: legGroups, tapLeg: legGroups[3] };
}

/** Живая муха: усики дрожат от того, что чуют, крылья машут тем сильнее, чем активнее выход в тело.
 *  orn и dn — средние частоты обонятельных рецепторов и нисходящих нейронов (или любые 0…1). */
export function animateFly({ antennae, wings }, { orn = 0, dn = 0, time = 0, minBuzz = 0 } = {}) {
  antennae.forEach((a, j) => { a.rotation.z = 0.25 + orn * 0.5 * Math.sin(time * 23 + j * 1.7); });
  const buzz = Math.max(minBuzz, Math.min(1, dn * 4));
  wings.forEach((w, j) => {
    w.rotation.x = -(j ? 1 : -1) * (0.12 + 0.55 * buzz * Math.abs(Math.sin(time * 40)));
  });
}
