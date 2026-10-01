// @ts-expect-error Node types are not in the browser app's tsconfig; Playwright runs this file in Node.
import { readFileSync, writeFileSync } from 'node:fs';
import { expect, test, type Page, type Request } from '@playwright/test';

test.use({ channel: 'chrome' });

// Relative to apps/web (Playwright's working directory). The fixture is a NON-personal stand-in: a re-fitted, re-posed
// CC0 MakeHuman body with a synthetic texture (see e2e/fixtures/twin-standin/README.md).
const fixture = 'e2e/fixtures/twin-standin';
const twinFiles = [`${fixture}/twin.glb`];
const shots = '../../docs/screenshots';

interface TwinProbe {
  vertexCount: number;
  vertex: (i: number) => [number, number, number];
  hiddenTriangles: () => number;
  handTriangles: () => number;
  handsVisible: () => boolean;
  handColor: () => string;
  repairedTexels: () => number;
  originalTexture: () => boolean;
  fingerVertices: () => number[][];
  pushDistances: () => number[];
  restPositions: () => number[];
  bodyVisible: () => boolean;
  twinVisible: () => boolean;
  alignment: () => { offset: number[]; maxResidual: number } | null;
  bone: (name: string) => { q: number[]; world: number[] } | null;
}
declare global {
  interface Window {
    __dtTwin?: TwinProbe;
    __dtParts?: { ids: () => string[] };
  }
}

async function fill(page: Page, testId: string, value: string): Promise<void> {
  await page.getByTestId(testId).fill(value);
}

async function fillRow(page: Page, measure: string, values: string[]): Promise<void> {
  for (const [index, value] of values.entries())
    await fill(page, `cell-${measure}-${index}`, value);
}

/** Skinned positions of 60 evenly spaced twin vertices (world space). */
async function samplePositions(page: Page): Promise<number[][]> {
  return page.evaluate(() => {
    const probe = window.__dtTwin!;
    const out: number[][] = [];
    for (let i = 0; i < 60; i++)
      out.push(probe.vertex(Math.floor((i * (probe.vertexCount - 1)) / 59)));
    return out;
  });
}

function maxDistance(a: number[][], b: number[][]): number {
  return Math.max(
    ...a.map((p, i) => Math.hypot(p[0]! - b[i]![0]!, p[1]! - b[i]![1]!, p[2]! - b[i]![2]!)),
  );
}

test('realistic twin: load the stand-in, pose it, dress it, switch back, remove it', async ({
  page,
}) => {
  test.setTimeout(300_000);
  const errors: string[] = [];
  const mutating: Request[] = [];
  const foreign: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error' && !message.location().url.endsWith('/favicon.ico'))
      errors.push(message.text());
  });
  page.on('request', (request) => {
    if (request.postData() || ['POST', 'PUT', 'PATCH'].includes(request.method()))
      mutating.push(request);
    if (!/^(http:\/\/localhost:5173|blob:|data:)/.test(request.url())) foreign.push(request.url());
  });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/');
  await expect(page.getByTestId('avatar-ready')).toBeAttached({ timeout: 120_000 });

  // --- the model switch starts on the standard mannequin; the twin needs its files first
  await expect(page.getByTestId('model-standard')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByTestId('model-twin')).toHaveAttribute('aria-pressed', 'false');
  await page.getByTestId('model-twin').click(); // no files yet: it opens the twin tab instead of switching
  await expect(page.getByTestId('twin-panel')).toBeVisible();
  await expect(page.getByTestId('model-standard')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByTestId('twin-status')).toHaveAttribute('data-status', 'empty');
  await expect(page.getByTestId('twin-privacy')).toContainText('yüklenmez');
  await expect
    .poll(() => page.evaluate(() => window.__dtParts?.ids().includes('eyes-default') ?? false), {
      timeout: 30_000,
    })
    .toBe(true); // standard mode has its parts

  // --- a file that is not a twin package is rejected with a readable message
  await page.getByTestId('twin-file-input').setInputFiles('e2e/fixtures/README.md');
  await expect(page.getByTestId('twin-status')).toHaveAttribute('data-status', 'error');
  await expect(page.getByTestId('twin-error')).toBeVisible();

  // --- pick one self-contained twin.glb: the twin replaces the mannequin
  await page.getByTestId('twin-file-input').setInputFiles(twinFiles);
  await expect(page.getByTestId('twin-status')).toHaveAttribute('data-status', 'ready');
  await expect(page.getByTestId('twin-status')).toHaveAttribute('data-active', 'true');
  await expect(page.getByTestId('model-twin')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByTestId('twin-runtime')).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId('twin-file-glb')).toContainText('twin.glb');
  await expect(page.getByTestId('twin-file-mapping')).toContainText('twin.glb içinde');
  await expect
    .poll(() => page.evaluate(() => window.__dtTwin?.twinVisible() ?? false), { timeout: 30_000 })
    .toBe(true);
  expect(await page.evaluate(() => window.__dtTwin!.bodyVisible())).toBe(false); // the MakeHuman body is solved but hidden
  expect(await page.evaluate(() => window.__dtTwin!.handsVisible())).toBe(true);
  expect(await page.evaluate(() => window.__dtTwin!.repairedTexels())).toBe(0);
  expect(await page.evaluate(() => window.__dtTwin!.originalTexture())).toBe(true);
  expect(await page.evaluate(() => window.__dtTwin!.handTriangles())).toBeGreaterThan(100);
  expect(await page.evaluate(() => window.__dtTwin!.handColor())).toBe('#d6a489');
  const handHidden = await page.evaluate(() => window.__dtTwin!.hiddenTriangles());
  expect(handHidden).toBeGreaterThan(100);
  const undressedPositions = await page.evaluate(() => window.__dtTwin!.restPositions());
  const alignment = await page.evaluate(() => window.__dtTwin!.alignment());
  expect(alignment?.maxResidual).toBeLessThan(1e-4); // twin.json and rigged.glb rebuild the same skeleton
  await expect.poll(() => page.evaluate(() => window.__dtParts?.ids().length ?? 0)).toBe(0); // eyes / hair are the twin's own

  // --- body sliders became read-only measurements from the scan; face and appearance are standard-mode features
  await page.getByRole('tab', { name: 'Ölçüler', exact: true }).click();
  await expect(page.getByTestId('twin-measurements')).toBeVisible();
  const expected = JSON.parse(readFileSync(`${fixture}/twin.json`, 'utf8') as string) as {
    measurementsCm: Record<string, number>;
  };
  await expect(page.getByTestId('twin-measure-height')).toContainText(
    expected.measurementsCm.height!.toFixed(1),
  );
  await expect(page.getByTestId('twin-measure-chest')).toContainText(
    expected.measurementsCm.chest!.toFixed(1),
  );
  await expect(page.getByTestId('twin-measurements-note')).toContainText('salt okunur');
  await expect(page.getByRole('spinbutton', { name: 'Göğüs çevresi', exact: true })).toHaveCount(0);
  await expect(page.getByTestId('twin-appearance-note')).toBeVisible();
  await page.getByRole('tab', { name: 'Yüz', exact: true }).click();
  await expect(page.getByTestId('twin-face-note')).toBeVisible();
  await page.getByRole('tab', { name: 'Ölçüler', exact: true }).click();

  // --- poses drive the same skeleton, so bones and twin vertices move together
  await page.getByRole('button', { name: 'T-Poz', exact: true }).click();
  await page.waitForTimeout(1200);
  const tBone = await page.evaluate(() => window.__dtTwin!.bone('upperarm_l')!);
  const tHand = await page.evaluate(() => window.__dtTwin!.bone('hand_l')!.world);
  const tSample = await samplePositions(page);
  const tFingers = await page.evaluate(() => window.__dtTwin!.fingerVertices());
  expect(tFingers.length).toBeGreaterThan(0);
  await page.getByRole('button', { name: 'Tam vücut', exact: true }).first().click();
  await page.waitForTimeout(2500);
  await page.screenshot({ path: `${shots}/twin-standin-tpose.png` });
  await page.getByRole('button', { name: 'Rahat', exact: true }).click();
  await page.waitForTimeout(1200);
  const rBone = await page.evaluate(() => window.__dtTwin!.bone('upperarm_l')!);
  const rHand = await page.evaluate(() => window.__dtTwin!.bone('hand_l')!.world);
  const rSample = await samplePositions(page);
  const rFingers = await page.evaluate(() => window.__dtTwin!.fingerVertices());
  expect(maxDistance(tFingers, rFingers)).toBeGreaterThan(0.001); // finger motion in the wrist frame
  expect(Math.hypot(...rBone.q.map((v, i) => v - tBone.q[i]!))).toBeGreaterThan(0.05); // the bone rotated
  expect(
    Math.hypot(rHand[0]! - tHand[0]!, rHand[1]! - tHand[1]!, rHand[2]! - tHand[2]!),
  ).toBeGreaterThan(0.2); // the hand followed
  expect(maxDistance(tSample, rSample)).toBeGreaterThan(0.2); // and so did the scan's own vertices
  await page.getByRole('button', { name: 'Yürüyüş', exact: true }).click();
  await page.waitForTimeout(1200);
  expect(maxDistance(rSample, await samplePositions(page))).toBeGreaterThan(0.1);
  await page.getByRole('button', { name: 'A-Poz', exact: true }).click();
  await page.waitForTimeout(1200);
  await page.screenshot({ path: `${shots}/twin-standin-apose.png` });

  // --- the camera presets are bone based: they follow the twin
  await page.getByRole('button', { name: 'Yüz', exact: true }).first().click();
  await page.waitForTimeout(2500);
  await page.screenshot({ path: `${shots}/twin-standin-face.png` });
  await page.getByRole('button', { name: 'Tam vücut', exact: true }).first().click();

  // --- wardrobe: a T-shirt bound to the hidden body renders over the twin and hides the scan under it
  await page.getByRole('tab', { name: 'Gardırop', exact: true }).click();
  await expect(page.getByTestId('worn-empty')).toBeVisible();
  expect(await page.evaluate(() => window.__dtTwin!.hiddenTriangles())).toBe(handHidden);
  await page.getByTestId('use-template-tshirt').click();
  await fill(page, 'form-name', 'Twin tişört');
  await fill(page, 'form-color-hex', '#3b6ea8');
  await fillRow(page, 'chest', ['98', '106', '114']);
  await fillRow(page, 'waist', ['90', '98', '106']);
  await fillRow(page, 'length', ['68', '70', '72']);
  await fillRow(page, 'sleeve', ['19', '20', '21']);
  await page.getByTestId('form-selected-size').selectOption('M');
  await page.getByTestId('form-save-wear').click();
  await expect(page.getByTestId('worn-top')).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId('fit-overall')).toBeVisible(); // sized against the twin's measurements
  await expect
    .poll(() => page.evaluate(() => window.__dtTwin!.hiddenTriangles()), { timeout: 30_000 })
    .toBeGreaterThan(500);
  await page.waitForTimeout(1500);
  const hiddenWithTop = await page.evaluate(() => window.__dtTwin!.hiddenTriangles());

  await page.getByTestId('wardrobe-add').click();
  await page.getByTestId('form-template').selectOption('jeans');
  await fill(page, 'form-name', 'Twin jean');
  await page.getByTestId('form-flat').check();
  await fillRow(page, 'waist', ['40', '43', '46']);
  await fillRow(page, 'hip', ['49', '52', '55']);
  await fillRow(page, 'inseam', ['78', '79', '80']);
  await fillRow(page, 'thigh', ['30', '31.5', '33']);
  await page.getByTestId('form-save-wear').click();
  await expect(page.getByTestId('worn-bottom')).toBeVisible({ timeout: 30_000 });
  await expect
    .poll(() => page.evaluate(() => window.__dtTwin!.hiddenTriangles()), { timeout: 30_000 })
    .toBeGreaterThan(hiddenWithTop + 500);
  await page.waitForTimeout(1500);
  const hiddenWithOutfit = await page.evaluate(() => window.__dtTwin!.hiddenTriangles());
  const push = await page.evaluate(() => window.__dtTwin!.pushDistances());
  expect(Math.max(...push)).toBeLessThanOrEqual(0.008);
  expect(push.filter((d) => d > 0).length).toBeGreaterThan(100);
  expect(push.some((d) => d > 0 && d < 0.008)).toBe(true); // smooth margin around neckline and waistband
  expect(await page.evaluate(() => window.__dtTwin!.handsVisible())).toBe(true);
  expect(await page.evaluate(() => window.__dtTwin!.repairedTexels())).toBeGreaterThan(0);
  expect(await page.evaluate(() => window.__dtTwin!.originalTexture())).toBe(false);
  await page.getByRole('tab', { name: 'Gerçekçi ikiz', exact: true }).click();
  await expect(page.getByTestId('twin-runtime')).toHaveAttribute(
    'data-hidden',
    String(hiddenWithOutfit),
  );
  await page.getByRole('tab', { name: 'Gardırop', exact: true }).click();

  // the outfit moves with the poses, the garments and the twin on one skeleton
  await page.getByRole('button', { name: 'T-Poz', exact: true }).click();
  await page.getByRole('button', { name: 'Tam vücut', exact: true }).first().click();
  await page.waitForTimeout(3000);
  await page.screenshot({ path: `${shots}/twin-standin-outfit.png` });
  await page.getByRole('button', { name: 'Yürüyüş', exact: true }).click();
  await page.waitForTimeout(2000);
  await page.screenshot({ path: `${shots}/twin-standin-walk.png` });

  // taking the clothes off shows the whole scan again
  await page.getByTestId('take-off-bottom').click();
  await page.getByTestId('take-off-top').click();
  await expect.poll(() => page.evaluate(() => window.__dtTwin!.hiddenTriangles())).toBe(handHidden);
  expect(await page.evaluate(() => window.__dtTwin!.pushDistances().every((d) => d === 0))).toBe(
    true,
  );
  expect(await page.evaluate(() => window.__dtTwin!.restPositions())).toEqual(undressedPositions);
  expect(await page.evaluate(() => window.__dtTwin!.repairedTexels())).toBe(0);
  expect(await page.evaluate(() => window.__dtTwin!.originalTexture())).toBe(true);

  // --- the twin survives a reload (IndexedDB, no upload) and is restored in twin mode
  await page.reload();
  await expect(page.getByTestId('avatar-ready')).toBeAttached({ timeout: 120_000 });
  await expect(page.getByTestId('model-twin')).toHaveAttribute('aria-pressed', 'true', {
    timeout: 30_000,
  });
  await expect
    .poll(() => page.evaluate(() => window.__dtTwin?.twinVisible() ?? false), { timeout: 60_000 })
    .toBe(true);

  // --- switch back to the standard mannequin: body, parts, sliders and face come back
  await page.getByTestId('model-standard').click();
  await expect(page.getByTestId('model-standard')).toHaveAttribute('aria-pressed', 'true');
  await expect.poll(() => page.evaluate(() => window.__dtTwin === undefined)).toBe(true);
  await expect
    .poll(() => page.evaluate(() => window.__dtParts?.ids().includes('eyes-default') ?? false), {
      timeout: 30_000,
    })
    .toBe(true);
  await expect(page.getByRole('spinbutton', { name: 'Göğüs çevresi', exact: true })).toBeVisible();
  await expect(page.getByTestId('twin-measurements')).toHaveCount(0);
  await page.getByRole('button', { name: 'Tam vücut', exact: true }).first().click();
  await page.waitForTimeout(3000);
  await page.screenshot({ path: `${shots}/twin-standin-back-to-standard.png` });
  // and forth again without reloading the files
  await page.getByTestId('model-twin').click();
  await expect(page.getByTestId('model-twin')).toHaveAttribute('aria-pressed', 'true');
  await expect
    .poll(() => page.evaluate(() => window.__dtTwin?.twinVisible() ?? false), { timeout: 30_000 })
    .toBe(true);

  // --- remove the twin: files are gone, the standard model stays
  await page.getByRole('tab', { name: 'Gerçekçi ikiz', exact: true }).click();
  await page.getByTestId('twin-panel').screenshot({ path: `${shots}/twin-standin-panel.png` });
  await page.getByTestId('twin-remove').click();
  await expect(page.getByTestId('twin-status')).toHaveAttribute('data-status', 'empty');
  await expect(page.getByTestId('model-standard')).toHaveAttribute('aria-pressed', 'true');
  await expect.poll(() => page.evaluate(() => window.__dtTwin === undefined)).toBe(true);
  await page.reload();
  await expect(page.getByTestId('avatar-ready')).toBeAttached({ timeout: 120_000 });
  await page.getByRole('tab', { name: 'Gerçekçi ikiz', exact: true }).click();
  await expect(page.getByTestId('twin-status')).toHaveAttribute('data-status', 'empty'); // nothing was kept

  // nothing was uploaded and nothing left the local server
  expect(mutating.map((r) => `${r.method()} ${r.url()}`)).toEqual([]);
  expect(foreign).toEqual([]);
  expect(errors).toEqual([]);
});

test('realistic twin: files that do not belong together are rejected, a missing mapping still works', async ({
  page,
}, testInfo) => {
  test.setTimeout(240_000);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    // the rejected twin is reported with console.error by the app; anything else is a failure
    if (
      message.type() === 'error' &&
      !message.location().url.endsWith('/favicon.ico') &&
      !message.text().includes('Could not show the twin')
    )
      errors.push(message.text());
  });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/');
  await expect(page.getByTestId('avatar-ready')).toBeAttached({ timeout: 120_000 });
  await page.getByRole('tab', { name: 'Gerçekçi ikiz', exact: true }).click();

  // a twin.json of another body (much taller fitted shape) does not rebuild the skeleton of this glb
  const twin = JSON.parse(readFileSync(`${fixture}/twin.json`, 'utf8') as string) as {
    fittedMacros: Record<string, number>;
  };
  twin.fittedMacros.height = 0.95;
  twin.fittedMacros.gender = 0.1;
  const foreign = testInfo.outputPath('twin.json');
  writeFileSync(foreign, JSON.stringify(twin));
  await page
    .getByTestId('twin-file-input')
    .setInputFiles([`${fixture}/rigged.glb`, foreign, `${fixture}/mh2twin.bin`]);
  await expect(page.getByTestId('twin-status')).toHaveAttribute('data-status', 'error', {
    timeout: 60_000,
  });
  await expect(page.getByTestId('twin-error')).toHaveAttribute('data-code', 'mismatch');
  await expect(page.getByTestId('model-standard')).toHaveAttribute('aria-pressed', 'true');
  await expect.poll(() => page.evaluate(() => window.__dtTwin === undefined)).toBe(true);
  await expect
    .poll(() => page.evaluate(() => window.__dtParts?.ids().includes('eyes-default') ?? false), {
      timeout: 30_000,
    })
    .toBe(true);

  // the right twin.json without the optional mapping: the twin shows, garments still hide the scan under them
  await page
    .getByTestId('twin-file-input')
    .setInputFiles([`${fixture}/rigged.glb`, `${fixture}/twin.json`]);
  await expect(page.getByTestId('twin-status')).toHaveAttribute('data-status', 'ready');
  await expect(page.getByTestId('twin-status')).toHaveAttribute('data-active', 'true');
  await expect(page.getByTestId('twin-no-mapping')).toBeVisible();
  await expect
    .poll(() => page.evaluate(() => window.__dtTwin?.twinVisible() ?? false), { timeout: 60_000 })
    .toBe(true);
  // Legacy packages have no skinToneHex: the forearm texture must provide the replacement hand colour.
  const legacyTone = await page.evaluate(() => window.__dtTwin!.handColor());
  expect(legacyTone).not.toBe('#ffffff');
  expect(legacyTone).not.toBe('#c99a7e'); // confirms sampling, rather than the neutral unavailable-texture fallback
  await page.getByRole('tab', { name: 'Gardırop', exact: true }).click();
  await page.getByTestId('use-template-tshirt').click();
  await fill(page, 'form-name', 'Twin tişört');
  await fillRow(page, 'chest', ['98', '106', '114']);
  await fillRow(page, 'waist', ['90', '98', '106']);
  await fillRow(page, 'length', ['68', '70', '72']);
  await fillRow(page, 'sleeve', ['19', '20', '21']);
  await page.getByTestId('form-selected-size').selectOption('M');
  await page.getByTestId('form-save-wear').click();
  await expect(page.getByTestId('worn-top')).toBeVisible({ timeout: 30_000 });
  await expect
    .poll(() => page.evaluate(() => window.__dtTwin!.hiddenTriangles()), { timeout: 30_000 })
    .toBeGreaterThan(300);
  await page.getByTestId('take-off-top').click();
  await page.getByRole('tab', { name: 'Gerçekçi ikiz', exact: true }).click();
  await page.getByTestId('twin-remove').click();
  expect(errors).toEqual([]);
});
