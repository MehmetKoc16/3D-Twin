import { expect, test, type Request } from '@playwright/test';
import { makeMockFaceMap } from '../src/features/face/bake/mockFaceMap';

test.use({ channel: 'chrome' });

// Relative to apps/web (Playwright's working directory).
const fixture = 'e2e/fixtures/portrait.jpg';

test('selfie: detect 478 landmarks in-browser, bake, and never upload the photo', async ({ page }) => {
  test.setTimeout(180_000);
  const errors: string[] = [];
  const mutating: Request[] = [];
  page.on('console', (m) => { if (m.text().includes('[face]')) console.log(m.text()); });
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('request', (request) => {
    if (request.postData() || ['POST', 'PUT', 'PATCH'].includes(request.method())) mutating.push(request);
  });

  // Use the real face map when the asset pipeline has produced it, otherwise a hand-made mock.
  let usedMock = false;
  await page.route('**/assets/body/face-map.json', async (route) => {
    const response = await route.fetch();
    const type = response.headers()['content-type'] ?? '';
    if (response.ok() && !type.includes('text/html')) return route.fulfill({ response });
    usedMock = true;
    return route.fulfill({ contentType: 'application/json', body: JSON.stringify(makeMockFaceMap()) });
  });

  await page.goto('/src/features/face/dev/harness.html');
  await expect(page.getByTestId('face-privacy')).toBeVisible();
  await page.getByTestId('face-file-input').setInputFiles(fixture);

  await expect(page.getByTestId('face-status')).toHaveAttribute('data-status', 'ready-to-bake', { timeout: 120_000 });
  await expect(page.getByTestId('face-landmark-count')).toHaveAttribute('data-count', '478');

  await page.getByTestId('face-apply').click();
  await expect(page.getByTestId('face-status')).toHaveAttribute('data-status', 'baked', { timeout: 60_000 });

  const info = await page.evaluate(() => {
    const store = (window as unknown as {
      __faceStore: { getState: () => { overlayCanvas?: HTMLCanvasElement; revision: number; skinToneHex?: string } };
    }).__faceStore;
    const state = store.getState();
    const canvas = state.overlayCanvas;
    let opaque = 0;
    if (canvas) {
      const data = canvas.getContext('2d')?.getImageData(0, 0, canvas.width, canvas.height).data;
      if (data) for (let i = 3; i < data.length; i += 4 * 97) if ((data[i] ?? 0) > 0) opaque += 1;
    }
    return { width: canvas?.width, height: canvas?.height, revision: state.revision, skin: state.skinToneHex, opaque };
  });
  expect(info.width).toBe(2048);
  expect(info.height).toBe(2048);
  expect(info.revision).toBe(1);
  expect(info.skin).toMatch(/^#[0-9a-f]{6}$/);
  expect(info.opaque).toBeGreaterThan(0);
  console.log(`face bake done (mock face map: ${usedMock}), skin ${info.skin}`);

  // Privacy: no request carried a body or used a mutating method.
  expect(mutating.map((r) => `${r.method()} ${r.url()}`)).toEqual([]);

  // Remove clears everything.
  await page.getByTestId('face-remove').click();
  await expect(page.getByTestId('face-status')).toHaveAttribute('data-status', 'idle');
  expect(errors).toEqual([]);
});
