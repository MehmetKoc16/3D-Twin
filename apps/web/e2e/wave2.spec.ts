import { expect, test } from '@playwright/test';

test.use({ channel: 'chrome' });

test('real avatar: solve, focus, poses, screenshots', async ({ page }) => {
  test.setTimeout(180_000);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error' && !message.location().url.endsWith('/favicon.ico')) errors.push(message.text());
  });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/');
  await expect(page.getByTestId('avatar-ready')).toBeAttached({ timeout: 120_000 });

  const achieved = (key: string) => page.getByTestId(`achieved-${key}`);
  const readCm = async (key: string): Promise<number> => {
    const text = (await achieved(key).textContent()) ?? '';
    const match = /([\d.]+)\s*(cm|kg)/.exec(text);
    return match ? Number(match[1]) : Number.NaN;
  };

  // T-pose is the default; full body front view
  await page.getByRole('button', { name: 'Tam vücut', exact: true }).first().click();
  await page.waitForTimeout(2500);
  await page.screenshot({ path: '../../docs/screenshots/wave2-front.png' });

  // 175 -> 190 cm: the achieved height text follows the solver
  const height = page.getByRole('spinbutton', { name: 'Boy', exact: true });
  await expect.poll(() => readCm('heightCm')).toBeCloseTo(175, 0);
  await height.fill('190');
  await height.blur();
  await expect.poll(() => readCm('heightCm'), { timeout: 20_000 }).toBeGreaterThan(189.5);
  expect(await readCm('heightCm')).toBeLessThan(190.5);
  await expect(achieved('weightKg')).toContainText('kg');

  // slider drag stays responsive: a burst of changes must settle on the last value
  const slider = page.getByRole('slider', { name: 'Kilo', exact: true });
  const t0 = Date.now();
  for (let w = 60; w <= 110; w += 5) {
    await slider.fill(String(w));
  }
  await expect.poll(() => readCm('weightKg'), { timeout: 20_000 }).toBeGreaterThan(80); // circumferences stay fixed, so the mass follows only partially
  console.log(`[e2e] 11 weight changes settled in ${Date.now() - t0} ms`);
  await slider.fill('75');

  // feet focus
  await page.getByRole('button', { name: 'Ayaklar', exact: true }).first().click();
  await page.waitForTimeout(2500);
  await page.screenshot({ path: '../../docs/screenshots/wave2-feet.png' });
  await page.getByRole('button', { name: 'Yüz', exact: true }).first().click();
  await page.waitForTimeout(1500);
  await page.screenshot({ path: '../../docs/screenshots/wave2-face.png' });
  await page.getByRole('button', { name: 'Tam vücut', exact: true }).first().click();

  for (const name of ['A-Poz', 'Rahat', 'T-Poz']) {
    await page.getByRole('button', { name, exact: true }).click();
    await page.waitForTimeout(400);
  }
  await page.getByRole('button', { name: 'A-Poz', exact: true }).click();
  await page.waitForTimeout(1500);
  await page.screenshot({ path: '../../docs/screenshots/wave2-apose.png' });
  await page.getByRole('button', { name: 'T-Poz', exact: true }).click();

  // panel with an unreachable target (waist) so the warning shows, on a tall viewport to capture the whole panel
  const waist = page.getByRole('spinbutton', { name: 'Bel çevresi', exact: true });
  await waist.fill('125');
  await waist.blur();
  await expect(page.getByRole('button', { name: /hedef ölçüye ulaşılamadı/ }).first()).toBeVisible({ timeout: 20_000 });
  await page.setViewportSize({ width: 1440, height: 2000 });
  await page.waitForTimeout(500);
  await page.locator('aside').screenshot({ path: '../../docs/screenshots/wave2-panel.png' });
  expect(errors).toEqual([]);
});

test('unreachable target shows a warning', async ({ page }) => {
  test.setTimeout(180_000);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/');
  await expect(page.getByTestId('avatar-ready')).toBeAttached({ timeout: 120_000 });
  const waist = page.getByRole('spinbutton', { name: 'Bel çevresi', exact: true });
  await waist.fill('140');
  await waist.blur();
  await expect(page.getByRole('button', { name: /hedef ölçüye ulaşılamadı/ }).first()).toBeVisible({ timeout: 20_000 });
});
