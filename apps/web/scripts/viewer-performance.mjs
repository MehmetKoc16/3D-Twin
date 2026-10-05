import { chromium } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

// Only the permissive, synthetic stand-in is loaded. Never reads personal twin data.
const output = resolve(process.argv[2] ?? 'test-results/viewer-performance');
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const page = await browser.newPage({
  viewport: { width: 1440, height: 900 },
  deviceScaleFactor: 1,
});
const errors = [];
const httpFailures = [];
page.on('response', (response) => {
  if (response.status() >= 400)
    httpFailures.push({
      url: response.url(),
      status: response.status(),
      resourceType: response.request().resourceType(),
    });
});
page.on('pageerror', (error) => errors.push(error.message));
page.on('console', (message) => {
  if (message.type() === 'error')
    errors.push(`${message.text()} [${message.location().url || 'no source URL'}]`);
});
try {
  await page.goto(process.env.VIEWER_URL ?? 'http://localhost:5173');
  await page.getByTestId('avatar-ready').waitFor({ state: 'attached', timeout: 120_000 });
  await page.getByRole('button', { name: 'Dili değiştir' }).click();
  const gpu = await page.locator('canvas').evaluate((canvas) => {
    const gl = canvas.getContext('webgl2');
    const info = gl?.getExtension('WEBGL_debug_renderer_info');
    return {
      renderer: info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : 'unavailable',
      drawingBuffer: [gl?.drawingBufferWidth, gl?.drawingBufferHeight],
    };
  });
  const results = [];
  for (const model of ['standard', 'twin']) {
    if (model === 'twin') {
      await page.getByTestId('model-twin').click();
      await page.getByTestId('twin-file-input').setInputFiles('e2e/fixtures/twin-standin/twin.glb');
      await page.getByTestId('twin-runtime').waitFor({ timeout: 60_000 });
    }
    const quality = page.getByTestId('viewer-quality');
    const levels = (await quality.count()) ? ['high', 'performance'] : ['baseline'];
    if (levels[0] !== 'baseline')
      await page
        .locator('details')
        .filter({ has: quality })
        .evaluate((element) => {
          element.open = true;
        });
    for (const level of levels) {
      if (level !== 'baseline') await quality.selectOption(level);
      await page.getByRole('button', { name: 'Auto rotate', exact: true }).click();
      await page.waitForTimeout(3000); // excludes loading, shader compilation and camera settling
      const samples = await page.evaluate(
        () =>
          new Promise((done) => {
            const intervals = [];
            let start;
            let previous;
            function frame(time) {
              start ??= time;
              if (previous !== undefined) intervals.push(time - previous);
              previous = time;
              if (time - start < 6000) globalThis.requestAnimationFrame(frame);
              else done(intervals);
            }
            globalThis.requestAnimationFrame(frame);
          }),
      );
      const sorted = [...samples].sort((a, b) => a - b);
      const averageMs = samples.reduce((sum, value) => sum + value, 0) / samples.length;
      results.push({
        model,
        quality: level,
        frames: samples.length,
        averageMs,
        p95Ms: sorted[Math.floor(sorted.length * 0.95)],
        fps: 1000 / averageMs,
      });
      await page.getByRole('button', { name: 'Auto rotate', exact: true }).click();
      await page.screenshot({ path: resolve(output, `${model}-${level}.png`) });
    }
  }
  const report = {
    browser: browser.version(),
    viewport: [1440, 900],
    gpu,
    results,
    errors,
    httpFailures,
    method:
      '6 seconds requestAnimationFrame intervals during CameraControls auto orbit, after 3 seconds warmup; vsync-limited, not GPU timer queries',
  };
  await writeFile(resolve(output, 'metrics.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
  if (errors.length || httpFailures.length) process.exitCode = 1;
} finally {
  await browser.close();
}
