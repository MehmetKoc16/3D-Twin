import { expect, test, type Request } from '@playwright/test';

test.use({ channel: 'chrome' });

// Relative to apps/web (Playwright's working directory).
const fixture = 'e2e/fixtures/portrait.jpg';

test('selfie on the real avatar: real face map, texture + shape fit, nothing uploaded', async ({ page }) => {
  test.setTimeout(240_000);
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
  const ready = page.getByTestId('avatar-ready');
  await expect(ready).toBeAttached({ timeout: 120_000 });
  await expect(ready).toHaveAttribute('data-face-revision', '0');

  await page.getByRole('tab', { name: 'Yüz', exact: true }).click();
  await expect(page.getByTestId('face-panel')).toBeVisible();
  await page.getByTestId('face-file-input').setInputFiles(fixture);
  await expect(page.getByTestId('face-status')).toHaveAttribute('data-status', 'ready-to-bake', { timeout: 120_000 });
  await page.getByTestId('face-apply').click();
  await expect(page.getByTestId('face-status')).toHaveAttribute('data-status', 'baked', { timeout: 60_000 });

  // the avatar picked up the baked face, and the worker fitted the face shape
  await expect(ready).toHaveAttribute('data-face-revision', '1', { timeout: 20_000 });
  await expect(page.getByTestId('face-fit')).toHaveAttribute('data-state', 'fitted', { timeout: 30_000 });
  const rms = Number(await page.getByTestId('face-fit').getAttribute('data-rms'));
  expect(rms).toBeGreaterThan(0);
  expect(rms).toBeLessThan(6);

  // pixel probe: the composite canvas inside the face UV bounds differs from the plain skin tone
  const probe = await page.evaluate(() => {
    const canvas = (window as unknown as { __dtSkinCanvas?: HTMLCanvasElement }).__dtSkinCanvas;
    if (!canvas) return null;
    const context = canvas.getContext('2d');
    if (!context) return null;
    const pixel = (u: number, v: number): number[] => Array.from(context.getImageData(Math.round(u * canvas.width), Math.round(v * canvas.height), 1, 1).data);
    const tone = pixel(0.1, 0.9); // outside the face region: plain skin tone
    let differing = 0;
    for (let i = 0; i < 25; i += 1) {
      const p = pixel(0.83 + (i % 5) * 0.025, 0.42 + Math.floor(i / 5) * 0.045);
      if (p.some((c, k) => Math.abs(c - (tone[k] ?? 0)) > 6)) differing += 1;
    }
    return { width: canvas.width, height: canvas.height, differing };
  });
  expect(probe).not.toBeNull();
  expect(probe?.width).toBe(2048);
  expect(probe?.differing).toBeGreaterThan(8);

  await page.getByRole('button', { name: 'Yüz', exact: true }).click();
  await page.waitForTimeout(3000);
  await page.screenshot({ path: '../../docs/screenshots/wave3-face-closeup.png' });
  await page.getByRole('button', { name: 'Tam vücut', exact: true }).first().click();
  await page.waitForTimeout(3000);
  await page.screenshot({ path: '../../docs/screenshots/wave3-face-full.png' });

  // privacy: no request carried a body or used a mutating method
  expect(mutating.map((r) => `${r.method()} ${r.url()}`)).toEqual([]);

  // removing the photo drops the face texture again
  await page.getByTestId('face-remove').click();
  await expect(ready).toHaveAttribute('data-face-revision', '0', { timeout: 20_000 });
  expect(errors).toEqual([]);
});
