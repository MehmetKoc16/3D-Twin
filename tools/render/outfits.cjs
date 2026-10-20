// Outfit render harness for the running dev server (http://localhost:5201): wears outfit combos on the twin
// (or the standard body with STD=1) and saves screenshots. Usage and env vars: see README.md in this folder.
const { createRequire } = require('module');
const req = createRequire(require('path').resolve(__dirname, '../../apps/web/package.json'));
const { chromium } = req('@playwright/test');
const path = require('path');
const fs = require('fs');
const OUT = process.env.OUT || require('path').join(__dirname, 'out');
require('fs').mkdirSync(OUT, { recursive: true });
const NATIVE = JSON.parse(fs.readFileSync(path.join(__dirname, 'native.json'), 'utf8'));
const GLB = process.env.GLB || path.resolve(__dirname, '../../user-data/twin/out/hybrid/twin.glb');
const errors = [];
const kombins = {
  1: ['mh-jacket-t01', 'mh-m-trousers-01', 'sf-sneakers-lowpoly'],
  2: ['mh-male-polo-shirt', 'mh-m-trousers-02', 'sf-dad-sneaker'],
  3: ['mh-jacket-t01', 'mh-denim-jacket', 'mh-m-trousers-01', 'sf-converse-style'],
  4: ['mh-male-polo-shirt', 'mh-denim-jacket', 'mh-m-trousers-02', 'sf-sneakers-lowpoly'],
  5: ['mh-jacket-t01', 'mh-m-trousers-02', 'mh-comfortable-sneakers'],
  6: ['mh-male-polo-shirt', 'mh-m-trousers-01', 'sf-converse-style'],
  7: ['mh-denim-jacket', 'mh-m-trousers-01', 'sf-converse-style'],
};
const ONLY = (process.env.ONLY || '1,2,3').split(',').map(Number);
const STD = process.env.STD === '1';
const PFX = process.env.PFX || '';
(async () => {
  let browser;
  try { browser = await chromium.launch({ channel: 'chrome' }); } catch { browser = await chromium.launch(); }
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await page.goto('http://localhost:5201/');
  await page.getByTestId('avatar-ready').waitFor({ state: 'attached', timeout: 120000 });
  if (!STD) {
  await page.getByTestId('model-twin').click();
  await page.getByRole('tab', { name: 'Gerçekçi ikiz', exact: true }).click();
  await page.getByTestId('twin-file-input').setInputFiles(GLB);
  await page.getByTestId('twin-runtime').waitFor({ timeout: 120000 });
  }
  if (process.env.BODY) {
    for (const [k, v] of Object.entries(JSON.parse(process.env.BODY))) {
      const box = page.locator(`#number-${k}`);
      if (await box.count()) { await box.fill(String(v)); await box.press('Tab'); await page.waitForTimeout(800); } else errors.push('no field ' + k);
    }
    await page.waitForTimeout(3000);
  }
  await page.getByRole('tab', { name: 'Gardırop', exact: true }).click();
  const create = async (id) => {
    await page.getByTestId('wardrobe-add').click();
    await page.getByTestId('form-template').selectOption(id);
    await page.getByTestId('form-name').fill(id);
    
    for (const [k, v] of Object.entries(/sneaker|^sf-/.test(id) ? {} : NATIVE[id])) {
      const f = [0.96, 1, 1.04];
      for (let i = 0; i < 3; i++) {
        const c = page.getByTestId(`cell-${k}-${i}`);
        if (await c.count()) await c.fill(String(+(v * f[i]).toFixed(1)));
      }
    }
    for (let i = 0; i < 12; i++) { const c = page.getByTestId('cell-footLength-' + i); if (false && (await c.count())) await c.fill(String(+(23.3 + 0.67 * (i - 1)).toFixed(1))); }
    await page.getByTestId('form-selected-size').selectOption({ index: 1 });
    await page.getByTestId('form-save-wear').click();
    await page.waitForTimeout(1200);
  };
  const all = [...new Set(ONLY.flatMap((n) => kombins[n]))];
  for (const id of all) {
    await create(id);
    const err = page.getByTestId('form-errors');
    if (await err.count()) { await page.screenshot({ path: path.join(OUT, 'dbg-' + id + '.png') }); errors.push('form error ' + id + ': ' + (await err.innerText())); await page.getByTestId('form-cancel').click(); }
  }
  const clearAll = async () => {
    for (const slot of ['top', 'bottom', 'shoes', 'outer']) {
      const b = page.getByTestId(`take-off-${slot}`);
      if (await b.count()) { await b.click(); await page.waitForTimeout(300); }
    }
  };
  const wear = async (id) => {
    await page.locator(`[data-testid^="item-"][data-name="${id}"] [data-testid^="wear-"]`).click();
    await page.waitForTimeout(600);
  };
  const focus = async (name) => { await page.getByRole('button', { name, exact: true }).first().click(); await page.waitForTimeout(1500); };
  const orbit = async (dx) => {
    await page.mouse.move(500, 450); await page.mouse.down(); await page.mouse.move(500 + dx, 450, { steps: 15 }); await page.mouse.up();
    await page.waitForTimeout(800);
  };
  const snap = async (name) => { await page.waitForTimeout(1200); await page.screenshot({ path: path.join(OUT, PFX + name), clip: { x: 0, y: 64, width: 1060, height: 836 } }); };
  for (const n of ONLY) {
    await clearAll();
    for (const id of kombins[n]) await wear(id);
    await page.waitForTimeout(3000);
    await page.getByRole('button', { name: 'A-Poz', exact: true }).click();
    await page.waitForTimeout(1500);
    await focus('Tam vücut'); await snap(`k${n}-apose-front.png`); await orbit(-60); await snap(`k${n}-apose-34.png`);
    if (process.env.UPPER === '1') { await focus('Üst vücut'); await snap(`k${n}-upper-front.png`); await orbit(-50); await snap(`k${n}-upper-34.png`); }
    if (process.env.POSES) { for (const p of process.env.POSES.split(',')) { await page.getByRole('button', { name: p, exact: true }).click(); await page.waitForTimeout(2500); await focus('Tam vücut'); await snap(`k${n}-${p.replace(/\s/g,'_')}-front.png`); await orbit(-60); await snap(`k${n}-${p.replace(/\s/g,'_')}-34.png`); } }
    if (process.env.FEET === '1') { await focus('Ayaklar'); await snap(`k${n}-feet-34.png`); await orbit(60); await snap(`k${n}-feet-front.png`); await orbit(120); await snap(`k${n}-feet-sideA.png`); await orbit(-240); await snap(`k${n}-feet-sideB.png`); }
  }
  console.log(errors.join('\n') || 'no errors');
  await browser.close();
})();
