import { expect, test } from '@playwright/test';

test.use({ channel: 'chrome' });

test('studio renders without shader errors and quality settings persist in both languages', async ({
  page,
}) => {
  const errors: string[] = [];
  const httpFailures: string[] = [];
  page.on('response', (response) => {
    if (response.status() >= 400) httpFailures.push(`${response.status()} ${response.url()}`);
  });
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(`${message.text()} [${message.location().url}]`);
  });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/');
  const icon = page.locator('head link[rel="icon"]');
  await expect(icon).toHaveCount(1);
  const iconUrl = await icon.getAttribute('href');
  expect(iconUrl).toBeTruthy();
  const iconResponse = await page.request.get(iconUrl!);
  expect(iconResponse.status()).toBe(200);
  expect(iconResponse.headers()['content-type']).toContain('image/svg+xml');
  await expect(page.getByTestId('avatar-ready')).toBeAttached({ timeout: 120_000 });
  await page.getByText('Görüntü ayarları', { exact: true }).click();
  const quality = page.getByTestId('viewer-quality');
  await expect(quality).toHaveValue('high');
  await quality.selectOption('performance');
  await page.reload();
  await page.getByText('Görüntü ayarları', { exact: true }).click();
  await expect(quality).toHaveValue('performance');
  await page.getByRole('button', { name: 'Dili değiştir' }).click();
  await expect(page.getByText('Display settings', { exact: true })).toBeVisible();
  await quality.selectOption('high');
  await page.getByRole('button', { name: 'Auto rotate', exact: true }).click();
  await page.waitForTimeout(2000);
  await page.screenshot({ path: 'test-results/viewer-studio.png' });
  expect(errors).toEqual([]);
  expect(httpFailures).toEqual([]);
});
