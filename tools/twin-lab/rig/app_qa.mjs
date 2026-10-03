// Visual QA of a twin package inside the web app (poses, zoom presets, wardrobe over the twin), with Playwright.
//
//   npm run dev -w @dt/web            # in another terminal (http://localhost:5173)
//   node tools/twin-lab/rig/app_qa.mjs <twinDir> <outDir> [--tag=twin] [--url=http://localhost:5173] [--no-wardrobe] [--standin] [--legacy]
//
// Prefer ../twin.glb or twin.glb; --legacy picks rigged.glb, twin.json (+ mh2twin.bin) instead.
// PRIVACY: for a real person keep <twinDir> and <outDir> under
// user-data/ (git-ignored), e.g. user-data/twin/out/rig -> user-data/twin/out/app. The files are handed to the app's
// file input like a user pick; nothing is uploaded (the script fails if the page makes a POST / PUT request).
import { existsSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '../../..');
const require = createRequire(path.join(repo, 'package.json'));
const { chromium } = require('playwright');

const positional = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const flags = Object.fromEntries(
  process.argv
    .slice(2)
    .filter((a) => a.startsWith('--'))
    .map((a) => {
      const [k, v = 'true'] = a.slice(2).split('=');
      return [k, v];
    }),
);
const [twinDir, outDir] = positional;
if (!twinDir || !outDir)
  throw new Error(
    'usage: app_qa.mjs <twinDir> <outDir> [--tag=twin] [--url=...] [--no-wardrobe] [--legacy]',
  );
// The screenshots show the twin's likeness: like shape/generate.py, refuse to write them anywhere in the repo outside
// user-data/ or a git-ignored .cache / outputs folder, unless the twin is the non-personal stand-in (--standin).
const outRel = path.relative(repo, path.resolve(outDir));
const outInsideRepo = !outRel.startsWith('..') && !path.isAbsolute(outRel);
const outIgnored = outRel
  .split(path.sep)
  .some((part) => ['user-data', '.cache', 'outputs'].includes(part));
if (outInsideRepo && !outIgnored && flags.standin === undefined)
  throw new Error(
    `refusing to write personal screenshots to tracked path ${outRel} (use user-data/..., or --standin for the stand-in)`,
  );
const tag = flags.tag ?? 'twin';
// A tag is a filename prefix, never a path: screenshots must stay inside the supplied outDir.
if (!/^[a-zA-Z0-9_-]+$/.test(tag)) throw new Error('tag must contain only letters, digits, _ or -');
const url = flags.url ?? 'http://localhost:5173';
mkdirSync(outDir, { recursive: true });
const bundle =
  flags.legacy === undefined
    ? [path.resolve(twinDir, '../twin.glb'), path.resolve(twinDir, 'twin.glb')].find(existsSync)
    : undefined;
const files = bundle
  ? [bundle]
  : ['rigged.glb', 'twin.json', 'mh2twin.bin']
      .map((f) => path.resolve(twinDir, f))
      .filter((f) => existsSync(f));
if (!bundle && !['rigged.glb', 'twin.json'].every((f) => existsSync(path.resolve(twinDir, f))))
  throw new Error('no twin bundle or complete legacy package found');

const browser = await chromium.launch({ channel: 'chrome' });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const problems = [];
page.on('pageerror', (e) =>
  problems.push(
    `pageerror: ${e.message} ${String(e.stack)
      .replace(/\s+at /g, ' | at ')
      .slice(0, 600)}`,
  ),
);
page.on('console', (m) => {
  if (m.type() === 'error' && !m.location().url.endsWith('/favicon.ico'))
    problems.push(`console: ${m.text().slice(0, 200)}`);
});
page.on('request', (r) => {
  if (['POST', 'PUT', 'PATCH'].includes(r.method()))
    problems.push(`mutating request: ${r.method()} ${r.url()}`);
});

const shot = async (name) => {
  await page.waitForTimeout(1800);
  await page.screenshot({ path: path.join(outDir, `${tag}-${name}.png`) });
  console.log('shot', name);
};
const pose = (label) => page.getByRole('button', { name: label, exact: true }).click();
const focus = (label) => page.getByRole('button', { name: label, exact: true }).first().click();
const orbit = async (dx) => {
  await page.mouse.move(520, 520);
  await page.mouse.down();
  await page.mouse.move(520 + dx, 520, { steps: 12 });
  await page.mouse.up();
  await page.waitForTimeout(800);
};
const fill = (id, v) => page.getByTestId(id).fill(v);
const row = async (m, vals) => {
  for (const [i, v] of vals.entries()) await fill(`cell-${m}-${i}`, v);
};

await page.goto(url);
await page.getByTestId('avatar-ready').waitFor({ state: 'attached', timeout: 120000 });
await page.getByRole('tab', { name: 'Gerçekçi ikiz' }).click();
await page.getByTestId('twin-file-input').setInputFiles(files);
await page.getByTestId('twin-runtime').waitFor({ timeout: 60000 });
console.log('runtime:', (await page.getByTestId('twin-runtime').innerText()).replace(/\n/g, ' | '));
console.log(
  'alignment:',
  JSON.stringify(await page.evaluate(() => window.__dtTwin?.alignment?.())),
);
await page.getByRole('tab', { name: 'Ölçüler', exact: true }).click();

// --- bare twin: poses and zoom presets
await pose('T-Poz');
await focus('Tam vücut');
await shot('bare-tpose');
await pose('Rahat');
await shot('bare-relaxed');
await pose('Yürüyüş');
await shot('bare-walk');
await pose('A-Poz');
await shot('bare-apose');
await focus('Yüz');
await shot('bare-face');
await focus('Üst vücut');
await shot('bare-upper');
await focus('Ayaklar');
await shot('bare-feet');

if (flags['no-wardrobe'] === undefined) {
  // --- wardrobe over the twin: T-shirt + jeans + sneakers
  await pose('Rahat');
  await focus('Tam vücut');
  await page.getByRole('tab', { name: 'Gardırop', exact: true }).click();
  await page.getByTestId('use-template-tshirt').click();
  await fill('form-name', 'QA tee');
  await fill('form-color-hex', '#3b6ea8');
  await row('chest', ['98', '106', '114']);
  await row('waist', ['90', '98', '106']);
  await row('length', ['68', '70', '72']);
  await row('sleeve', ['19', '20', '21']);
  await page.getByTestId('form-selected-size').selectOption('M');
  await page.getByTestId('form-save-wear').click();
  await page.getByTestId('worn-top').waitFor({ timeout: 60000 });
  await shot('tee-relaxed');
  console.log(
    'hidden twin triangles (tee):',
    await page.evaluate(() => window.__dtTwin?.hiddenTriangles?.()),
  );

  await page.getByTestId('wardrobe-add').click();
  await page.getByTestId('form-template').selectOption('jeans');
  await fill('form-name', 'QA jeans');
  await page.getByTestId('form-flat').check();
  await row('waist', ['40', '43', '46']);
  await row('hip', ['49', '52', '55']);
  await row('inseam', ['78', '79', '80']);
  await row('thigh', ['30', '31.5', '33']);
  await page.getByTestId('form-save-wear').click();
  await page.getByTestId('worn-bottom').waitFor({ timeout: 60000 });
  await shot('outfit-relaxed');
  console.log(
    'hidden twin triangles (tee + jeans):',
    await page.evaluate(() => window.__dtTwin?.hiddenTriangles?.()),
  );

  await page.getByTestId('wardrobe-add').click();
  await page.getByTestId('form-template').selectOption('sneakers');
  await fill('form-name', 'QA sneakers');
  await page.getByTestId('form-selected-size').selectOption('42');
  await page.getByTestId('form-save-wear').click();
  await page.getByTestId('worn-shoes').waitFor({ timeout: 60000 });
  await shot('outfit3-relaxed');
  console.log(
    'hidden twin triangles (+ shoes):',
    await page.evaluate(() => window.__dtTwin?.hiddenTriangles?.()),
  );

  await pose('T-Poz');
  await shot('outfit-tpose');
  await pose('Yürüyüş');
  await shot('outfit-walk');
  await pose('Eller belde');
  await shot('outfit-hips');
  await pose('Rahat');
  await focus('Üst vücut');
  await shot('outfit-upper');
  await orbit(220);
  await shot('outfit-upper-side');
  await focus('Alt vücut');
  await shot('outfit-lower');
  await focus('Ayaklar');
  await shot('outfit-feet');
  await focus('Tam vücut');
  await orbit(-430);
  await shot('outfit-back');
  await orbit(220);
  await shot('outfit-threequarter');

  // --- take everything off again: the bare twin is back
  for (const cat of ['shoes', 'bottom', 'top']) await page.getByTestId(`take-off-${cat}`).click();
  await page.waitForTimeout(1200);
  console.log(
    'hidden twin triangles (bare again):',
    await page.evaluate(() => window.__dtTwin?.hiddenTriangles?.()),
  );
}
console.log(
  problems.length ? `PROBLEMS:\n${problems.join('\n')}` : 'no page errors, nothing uploaded',
);
await browser.close();
if (problems.some((p) => p.startsWith('mutating'))) process.exit(1);
