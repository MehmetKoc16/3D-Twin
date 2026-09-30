import { expect, test, type Locator, type Page, type Request } from '@playwright/test';

test.use({ channel: 'chrome' });

// Relative to apps/web (Playwright's working directory).
const fixture = 'e2e/fixtures/portrait.jpg';
const shots = '../../docs/screenshots';

type Shot = Awaited<ReturnType<Locator['screenshot']>>;

interface PartsProbe {
  ids: () => string[];
  eyeCanvas: () => HTMLCanvasElement | null;
  hiddenBodyVertices: () => number;
}

async function partIds(page: Page): Promise<string[]> {
  return page.evaluate(() => (window as unknown as { __dtParts?: PartsProbe }).__dtParts?.ids() ?? []);
}

/** Iris colour (average over the iris ring) of the recoloured eye texture, as [r, g, b]. */
async function irisColor(page: Page): Promise<number[]> {
  return page.evaluate(() => {
    const canvas = (window as unknown as { __dtParts?: PartsProbe }).__dtParts?.eyeCanvas();
    if (!canvas) return [];
    const context = canvas.getContext('2d');
    if (!context) return [];
    const cx = 0.4904 * canvas.width;
    const cy = 0.4937 * canvas.height;
    const radius = 0.2041 * canvas.width;
    const data = context.getImageData(0, 0, canvas.width, canvas.height).data;
    const sum = [0, 0, 0];
    let n = 0;
    for (let y = 0; y < canvas.height; y++)
      for (let x = 0; x < canvas.width; x++) {
        const d = Math.hypot(x + 0.5 - cx, y + 0.5 - cy) / radius;
        if (d < 0.4 || d > 0.9) continue;
        for (let k = 0; k < 3; k++) sum[k] = (sum[k] ?? 0) + (data[(y * canvas.width + x) * 4 + k] ?? 0);
        n++;
      }
    return sum.map((v) => Math.round(v / n));
  });
}

async function focus(page: Page, name: string): Promise<void> {
  await page.getByRole('button', { name, exact: true }).click();
  await page.waitForTimeout(2600); // camera transition
}

/** Puts three viewer screenshots side by side in one image. */
async function sideBySide(page: Page, images: Shot[], path: string): Promise<void> {
  const composite = await page.context().newPage();
  const urls = images.map((image) => `data:image/png;base64,${image.toString('base64')}`);
  await composite.setContent(
    `<body style="margin:0;background:#171d21;display:flex">${urls.map((u) => `<img src="${u}" style="width:${Math.floor(1800 / urls.length)}px;height:auto">`).join('')}</body>`,
  );
  await composite.setViewportSize({ width: 1800, height: 700 });
  await composite.waitForFunction(() => [...document.images].every((img) => img.complete));
  await composite.screenshot({ path, fullPage: true });
  await composite.close();
}

async function pressed(locator: Locator): Promise<void> {
  await expect(locator).toHaveAttribute('aria-pressed', 'true');
}

test('appearance: hair, eyebrows, iris colour, colour from the photo, persistence', async ({ page }) => {
  test.setTimeout(300_000);
  const errors: string[] = [];
  const mutating: Request[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    // MediaPipe logs its TFLite delegate notice through console.error
    if (message.type() === 'error' && !message.text().startsWith('INFO:') && !message.location().url.endsWith('/favicon.ico')) errors.push(message.text());
  });
  page.on('request', (request) => {
    if (request.postData() || ['POST', 'PUT', 'PATCH'].includes(request.method())) mutating.push(request);
  });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/');
  await expect(page.getByTestId('avatar-ready')).toBeAttached({ timeout: 120_000 });

  // --- defaults: eyes, eyebrows, eyelashes, no hair; the eye-socket cavity is hidden
  await expect.poll(() => partIds(page), { timeout: 30_000 }).toEqual(expect.arrayContaining(['eyes-default', 'eyebrows-default', 'eyelashes-default']));
  expect(await partIds(page)).not.toContain('hair-long');
  expect(await page.evaluate(() => (window as unknown as { __dtParts: PartsProbe }).__dtParts.hiddenBodyVertices())).toBeGreaterThan(300);

  // --- appearance controls (measurements tab, bottom of the panel)
  const panel = page.getByTestId('parts-appearance');
  await panel.scrollIntoViewIfNeeded();
  await expect(panel).toBeVisible();
  await pressed(page.getByTestId('hair-style-none'));
  await pressed(page.getByTestId('brow-style-eyebrows-default'));

  const before = await irisColor(page);
  await page.getByTestId('hair-style-hair-long').click();
  await page.getByTestId('hair-color-red').click();
  await page.getByTestId('eye-color-green').click();
  await pressed(page.getByTestId('hair-style-hair-long'));
  await pressed(page.getByTestId('hair-color-red'));
  await pressed(page.getByTestId('eye-color-green'));
  await expect.poll(() => partIds(page)).toContain('hair-long');
  const green = await irisColor(page);
  expect(green[1]).toBeGreaterThan(green[0] ?? 0); // green channel above red
  expect(green).not.toEqual(before);

  // thick brows follow a hair colour change; "same as hair" is on by default
  await page.getByTestId('brow-style-eyebrows-thick').click();
  await expect.poll(() => partIds(page)).toContain('eyebrows-thick');
  expect(await partIds(page)).not.toContain('eyebrows-default');
  await expect(page.getByTestId('brow-follow-hair')).toBeChecked();

  // --- three hair styles, face view
  const frames: Shot[] = [];
  await focus(page, 'Yüz');
  for (const [style, color] of [
    ['hair-short', 'darkBrown'],
    ['hair-bob', 'blonde'],
    ['hair-ponytail', 'black'],
  ] as const) {
    await page.getByTestId(`hair-style-${style}`).click();
    await page.getByTestId(`hair-color-${color}`).click();
    await expect.poll(() => partIds(page)).toContain(style);
    await page.waitForTimeout(700);
    frames.push(await page.locator('main').screenshot());
  }
  await sideBySide(page, frames, `${shots}/wave3e-hair-styles.png`);

  // --- selfie: upload, bake, then take the eye colour from the photo
  await page.getByTestId('hair-style-hair-medium').click();
  await page.getByTestId('hair-color-brown').click();
  await page.getByRole('tab', { name: 'Yüz', exact: true }).click();
  await page.getByTestId('face-file-input').setInputFiles(fixture);
  await expect(page.getByTestId('face-status')).toHaveAttribute('data-status', 'ready-to-bake', { timeout: 120_000 });
  await page.getByTestId('face-apply').click();
  await expect(page.getByTestId('face-status')).toHaveAttribute('data-status', 'baked', { timeout: 60_000 });
  await expect(page.getByTestId('face-fit')).toHaveAttribute('data-state', 'fitted', { timeout: 30_000 });

  await page.getByRole('tab', { name: 'Ölçüler', exact: true }).click();
  await page.getByTestId('parts-appearance').scrollIntoViewIfNeeded();
  await expect(page.getByTestId('photo-colors')).toHaveAttribute('data-ready', 'true', { timeout: 30_000 });
  const fromPhoto = page.getByTestId('eye-from-photo');
  await expect(fromPhoto).toBeEnabled();
  await fromPhoto.click();
  const measured = await page.getByTestId('eye-color-custom').inputValue();
  expect(measured).toMatch(/^#[0-9a-f]{6}$/);
  expect(measured).not.toBe('#3f7a4a'); // no longer the green preset
  const fromPhotoColor = await irisColor(page);
  expect(fromPhotoColor.length).toBe(3);
  console.log(`eye colour from photo: ${measured}, iris ring average rgb(${fromPhotoColor.join(', ')})`);
  await page.getByTestId('hair-from-photo').click();

  // --- face close-up and full body
  await focus(page, 'Yüz');
  await page.locator('main').screenshot({ path: `${shots}/wave3e-face-closeup.png` });
  await focus(page, 'Tam vücut');
  await page.locator('main').screenshot({ path: `${shots}/wave3e-full.png` });

  // --- persistence: a reload restores hair, colours and eye colour
  await page.reload();
  await expect(page.getByTestId('avatar-ready')).toBeAttached({ timeout: 120_000 });
  await page.getByTestId('parts-appearance').scrollIntoViewIfNeeded();
  await pressed(page.getByTestId('hair-style-hair-medium'));
  await expect(page.getByTestId('eye-color-custom')).toHaveValue(measured);
  await expect.poll(() => partIds(page), { timeout: 30_000 }).toContain('hair-medium');

  // switching hair off removes the mesh again
  await page.getByTestId('hair-style-none').click();
  await expect.poll(() => partIds(page)).not.toContain('hair-medium');

  expect(mutating.map((r) => `${r.method()} ${r.url()}`)).toEqual([]); // nothing left the browser
  expect(errors).toEqual([]);
});
