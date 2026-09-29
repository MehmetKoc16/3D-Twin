import { expect, test } from '@playwright/test';

test.use({ channel: 'chrome' });

test('measurement, focus, pose, language, and screenshot', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error' && !message.location().url.endsWith('/favicon.ico')) errors.push(message.text());
  });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Dijital İkiz' })).toBeVisible();
  await expect(page.locator('canvas')).toBeVisible();
  const height = page.getByRole('spinbutton', { name: 'Boy', exact: true });
  await height.fill('185');
  await height.blur();
  await expect(height).toHaveValue('185');
  for (const name of ['Tam vücut', 'Yüz', 'Üst vücut', 'Alt vücut', 'Ayaklar']) {
    await page.getByRole('button', { name, exact: true }).first().click();
  }
  for (const name of ['T-Poz', 'A-Poz', 'Rahat', 'Eller belde', 'Yürüyüş', 'Yan profil']) {
    await page.getByRole('button', { name, exact: true }).click();
  }
  await page.getByRole('button', { name: 'Dili değiştir' }).click();
  await expect(page.getByRole('heading', { name: 'Body measurements' })).toBeVisible();
  await page.getByRole('button', { name: 'Change language' }).click();
  await page.getByRole('button', { name: 'Tam vücut' }).click();
  await page.getByRole('button', { name: 'T-Poz' }).click();
  await expect(page.getByRole('heading', { name: 'Vücut ölçüleri' })).toBeVisible();
  await page.waitForTimeout(800);
  await page.screenshot({ path: '../../docs/screenshots/wave1-ui.png' });
  await page.setViewportSize({ width: 800, height: 900 });
  const viewportBox = await page.getByRole('main').boundingBox();
  const panelBox = await page.getByRole('complementary').boundingBox();
  expect(viewportBox && panelBox && panelBox.y >= viewportBox.y + viewportBox.height).toBe(true);
  expect(errors).toEqual([]);
});
